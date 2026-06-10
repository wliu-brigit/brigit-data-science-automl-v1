"""Rule discovery from a logged anomaly trial — read-only, rerunnable.

Given an MLflow run id, this loads everything from what the harness already
logged (model, test predictions, dataset record) and runs the discovery
pipeline:

  1. attribution   — neutralize each feature family (replace with train
                     medians, re-score) to see what drives each row's anomaly
  2. residual queue — rows ranked by anomaly *after* neutralizing the
                     circular family (the heuristic's own inputs), i.e. the
                     anomaly the current rule-set cannot explain
  3. surrogate rules — a shallow decision tree distills the residual score
                     into explicit threshold conjunctions (candidate rules)
  4. enrichment    — each candidate rule scored against the fraud-flavored
                     outcome (gross DPD45 and never repaid as of snapshot),
                     band-split so heuristic circularity cannot flatter it

Usage:

    uv run python -m projects.fraud_anomaly_detection.analysis.rule_discovery \
        <run_id> [--top-frac 0.005] [--max-depth 3] [--min-leaf 50]

Read-only: nothing is written to MLflow or GCS.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from automl.mlflow import client as mlflow_client
from automl.mlflow.trial.artifacts import load_model
from automl.project import use_project

logging.getLogger("mlflow").setLevel(logging.ERROR)

UNIQUE_KEY = "advance_id"
BAND_COL = "heuristic_fraud_band"
BAND_ORDER = ["LOW", "POSSIBLE", "LIKELY", "EXTREMELY_LIKELY"]

# The heuristic's own inputs plus every alias/derivative of the same
# underlying count (network_* are SQL aliases of the lifetime count;
# 7d/30d/90d are window-supersets of 72h). Neutralizing this family asks:
# "what is anomalous about this row for reasons the current heuristic
# cannot already see?"
CIRCULAR_FAMILY = [
    "users_on_bank_account_72h",
    "users_on_bank_account_7d",
    "users_on_bank_account_30d",
    "users_on_bank_account_90d",
    "users_on_bank_account_lifetime_asof",
    "flag_3_users_on_bank_account_72h",
    "flag_5_users_on_bank_account_ever_asof",
    "avg_users_created_per_day_asof",
    "avg_users_created_per_month_asof",
    "user_creation_days_span_asof",
    "network_user_count_asof",
    "network_score_asof",
    "prior_advances_on_bank_account_72h",
    "prior_advances_on_bank_account_7d",
]

# Coarser families for per-row attribution reporting.
ATTRIBUTION_FAMILIES = {
    "bank_account_users(circular)": CIRCULAR_FAMILY,
    "prior_advance_velocity": [
        "prior_advances_on_bank_account_24h",
        "prior_advances_on_bank_account_30d",
        "prior_advances_on_bank_account_lifetime",
        "avg_prior_advances_per_day",
        "avg_prior_advances_per_month",
        "prior_advance_days_span",
        "hours_since_previous_advance_on_account",
    ],
    "amounts": [
        "loan_amount",
        "total_disbursed",
        "prior_loan_amount_avg_30d",
        "prior_loan_amount_sum_30d",
        "prior_total_disbursed_sum_30d",
        "express_transfer_fee",
    ],
    "identity_age_kyc": [
        "days_since_identity_created",
        "days_since_plaid_account_created",
        "days_between_identity_and_bank_account_creation",
        "hours_since_socure_created",
        "has_kyc",
        "bank_accounts_per_user_asof",
    ],
    "device_ip": [
        "has_device_id",
        "has_ip_address",
        "has_persistent_account_id",
        "signup_ip_matches_latest_ip",
    ],
    "timing": [
        "origination_hour",
        "origination_day_of_week",
        "is_weekend_origination",
    ],
}


@dataclass
class TrialContext:
    """Everything the pipeline needs, resolved from one MLflow run id."""

    run_id: str
    model: object
    test: pd.DataFrame   # test split + y_pred merged
    train: pd.DataFrame  # train split (median source for neutralization)
    model_features: list[str]
    dataset_id: str = ""
    notes: list[str] = field(default_factory=list)


def _download_runs_uri(uri: str) -> str:
    """Resolve a ``runs:/<run_id>/<path>`` artifact URI to a local file path."""
    run_id, artifact_path = uri.removeprefix("runs:/").split("/", 1)
    return mlflow_client.download_artifact(run_id, artifact_path, required=True)


def load_trial(run_id: str) -> TrialContext:
    """Resolve model, predictions, and data purely from the logged run.

    Requires an active project session (``use_project``) so the MLflow seam
    is bound — ``main()`` handles that for CLI use.
    """
    run = mlflow_client.raw().get_run(run_id)
    tags = run.data.tags

    record = json.load(open(_download_runs_uri(tags["data.record_uri"])))
    df = pd.read_parquet(record["data_gcs_uri"])
    split_col = record.get("split_pct_col", "SPLIT_PCT")
    test = df[df[split_col] >= 80].copy()
    train = df[df[split_col] < 80].copy()

    pred = pd.read_parquet(tags["eval.test.predictions_uri"])
    score_col = [c for c in pred.columns if c != UNIQUE_KEY and c != "is_fraud"][0]
    test = test.merge(
        pred[[UNIQUE_KEY, score_col]].rename(columns={score_col: "anomaly_score"}),
        on=UNIQUE_KEY,
        how="inner",
    )

    reg = pd.read_csv(
        mlflow_client.download_artifact(
            run_id, "features/dataset_feature_registry.csv", required=True
        )
    )
    model_features = reg[reg.model == True].name.tolist()  # noqa: E712

    model = load_model(run_id)
    return TrialContext(
        run_id=run_id,
        model=model,
        test=test,
        train=train,
        model_features=model_features,
        dataset_id=record.get("id", ""),
    )


def neutralized_scores(ctx: TrialContext, cols: list[str]) -> np.ndarray:
    """Re-score test with `cols` replaced by train medians (numeric) / modes."""
    present = [c for c in cols if c in ctx.test.columns]
    neutral = ctx.test.copy()
    for c in present:
        train_col = ctx.train[c]
        if pd.api.types.is_numeric_dtype(train_col):
            fill = train_col.median()
        else:
            mode = train_col.mode(dropna=True)
            fill = mode.iloc[0] if len(mode) else None
        neutral[c] = fill
    return np.asarray(ctx.model.predict(neutral)).ravel()


def attribution(ctx: TrialContext, families: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """Per-row anomaly attribution: score drop when each family is neutralized."""
    families = families or ATTRIBUTION_FAMILIES
    out = pd.DataFrame({UNIQUE_KEY: ctx.test[UNIQUE_KEY], "score": ctx.test["anomaly_score"]})
    for name, cols in families.items():
        out[f"drop:{name}"] = ctx.test["anomaly_score"].to_numpy() - neutralized_scores(ctx, cols)
    drop_cols = [c for c in out.columns if c.startswith("drop:")]
    out["dominant_family"] = (
        out[drop_cols].idxmax(axis=1).str.removeprefix("drop:").where(out[drop_cols].max(axis=1) > 0, "none")
    )
    return out


def residual_queue(ctx: TrialContext, top_frac: float = 0.005) -> pd.DataFrame:
    """Test rows ranked by anomaly after neutralizing the circular family."""
    resid = neutralized_scores(ctx, CIRCULAR_FAMILY)
    test = ctx.test.copy()
    test["residual_score"] = resid
    test = test.sort_values("residual_score", ascending=False)
    return test.head(max(1, int(len(test) * top_frac)))


def fraud_flavored_outcome(df: pd.DataFrame) -> pd.Series:
    """Gross DPD45 *and* never repaid as of snapshot — the bust-out cut.

    Only meaningful on mature rows (label_mature_d45 == 1); callers should
    filter first. Late-repaid DPD45 reads as credit risk, not fraud.
    """
    return (df["label_gross_dpd45"] == 1) & (df["label_repaid_current_snapshot"] == 0)


def surrogate_rules(
    ctx: TrialContext,
    residual: np.ndarray,
    max_depth: int = 3,
    min_leaf: int = 50,
) -> list[dict]:
    """Distill the residual score into threshold rules via a shallow tree."""
    from sklearn.tree import DecisionTreeRegressor

    feats = [
        c
        for c in ctx.model_features
        if c not in CIRCULAR_FAMILY and pd.api.types.is_numeric_dtype(ctx.test[c])
    ]
    X = ctx.test[feats].astype(float).fillna(ctx.train[feats].astype(float).median())
    tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_leaf, random_state=0)
    tree.fit(X, residual)

    t = tree.tree_
    rules: list[dict] = []

    def walk(node: int, conds: list[str], mask: np.ndarray) -> None:
        if t.children_left[node] == -1:  # leaf
            rules.append(
                {
                    "conditions": list(conds),
                    "n": int(mask.sum()),
                    "mean_residual_score": float(residual[mask].mean()),
                }
            )
            return
        f, thr = feats[t.feature[node]], t.threshold[node]
        left = X[f].to_numpy() <= thr
        walk(t.children_left[node], conds + [f"{f} <= {thr:.4g}"], mask & left)
        walk(t.children_right[node], conds + [f"{f} > {thr:.4g}"], mask & ~left)

    walk(0, [], np.ones(len(X), dtype=bool))
    rules.sort(key=lambda r: r["mean_residual_score"], reverse=True)

    # attach the row mask for downstream enrichment
    for rule in rules:
        mask = np.ones(len(X), dtype=bool)
        for cond in rule["conditions"]:
            col, op, val = cond.rsplit(" ", 2)[0], cond.split(" ")[-2], float(cond.split(" ")[-1])
            mask &= (X[col].to_numpy() <= val) if op == "<=" else (X[col].to_numpy() > val)
        rule["mask"] = mask
    return rules


def rule_enrichment(ctx: TrialContext, mask: np.ndarray) -> dict:
    """Support, fraud-flavored outcome rate, and band mix for one rule."""
    rows = ctx.test[mask]
    mature = rows[rows["label_mature_d45"] == 1]
    all_mature = ctx.test[ctx.test["label_mature_d45"] == 1]
    base = fraud_flavored_outcome(all_mature).mean()
    rate = fraud_flavored_outcome(mature).mean() if len(mature) else float("nan")
    bands = rows[BAND_COL].value_counts().to_dict()
    return {
        "n": int(len(rows)),
        "n_mature": int(len(mature)),
        "outcome_rate": float(rate),
        "outcome_base": float(base),
        "lift": float(rate / base) if base and len(mature) else float("nan"),
        "bands": {b: int(bands.get(b, 0)) for b in BAND_ORDER if bands.get(b)},
    }


def run_report(run_id: str, top_frac: float = 0.005, max_depth: int = 3, min_leaf: int = 50) -> None:
    ctx = load_trial(run_id)
    print(f"run {run_id} · dataset {ctx.dataset_id} · test rows {len(ctx.test)}")

    print("\n== 1. attribution: dominant family among the top 1% by raw score ==")
    att = attribution(ctx)
    top_raw = att.sort_values("score", ascending=False).head(max(1, len(att) // 100))
    print(top_raw["dominant_family"].value_counts().to_string())

    print(f"\n== 2. residual queue (top {top_frac:.1%} after neutralizing circular family) ==")
    queue = residual_queue(ctx, top_frac=top_frac)
    print(f"n={len(queue)} · band mix: {queue[BAND_COL].value_counts().to_dict()}")
    mature_q = queue[queue["label_mature_d45"] == 1]
    if len(mature_q):
        base = fraud_flavored_outcome(ctx.test[ctx.test["label_mature_d45"] == 1]).mean()
        rate = fraud_flavored_outcome(mature_q).mean()
        print(
            f"fraud-flavored outcome (DPD45 & never repaid): {rate:.2%} "
            f"vs base {base:.2%} → lift {rate / base:.1f}x (n_mature={len(mature_q)})"
        )

    print(f"\n== 3. surrogate rules on residual score (depth {max_depth}, min_leaf {min_leaf}) ==")
    resid_full = neutralized_scores(ctx, CIRCULAR_FAMILY)
    rules = surrogate_rules(ctx, resid_full, max_depth=max_depth, min_leaf=min_leaf)
    for i, rule in enumerate(rules[:5], 1):
        enr = rule_enrichment(ctx, rule["mask"])
        print(f"\nrule {i}: " + " AND ".join(rule["conditions"]))
        print(
            f"  n={enr['n']} (mature {enr['n_mature']}) · mean residual {rule['mean_residual_score']:.4f}"
        )
        print(
            f"  outcome {enr['outcome_rate']:.2%} vs base {enr['outcome_base']:.2%} "
            f"→ lift {enr['lift']:.1f}x · bands {enr['bands']}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_id")
    ap.add_argument("--top-frac", type=float, default=0.005)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--min-leaf", type=int, default=50)
    ap.add_argument("--full", action="store_true", help="full-run namespace (default: dry-run)")
    args = ap.parse_args()
    use_project("fraud_anomaly_detection", dry_run=not args.full)
    run_report(args.run_id, top_frac=args.top_frac, max_depth=args.max_depth, min_leaf=args.min_leaf)


if __name__ == "__main__":
    main()
