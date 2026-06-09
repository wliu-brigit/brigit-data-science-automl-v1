"""Next-layer conjunction discovery on the POST-LOCK residual (v1_76d3ad45).

The iterative loop (wendao): once the high-precision edges are locked into the
register, exclude that population and re-run discovery on what is LEFT, to find
the next layer. residual_mask() reads the register live, so this automatically
runs on the residual after the five currently-registered scenarios are gated out
(the two bank-account rings + the three 2026-06-08 edges).

Label spec (wendao): a supervised tree, so it needs matured labels — target
gross DPD45 on mature rows only (label_mature_d45 == 1; non-mature excluded, the
supervised-vs-unsupervised distinction). For every discovered leaf we report
BOTH DPD45 (credit + fraud) AND never-paid (fraud-leaning) so the "smell of
fraud" is visible: a leaf with high DPD45 but low never-paid is credit stress,
not a ring. Fit on TRAIN, precision VALIDATED on held-out TEST.

    uv run python -m projects.fraud_anomaly_detection.analysis.residual_next_layer
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_ID = "v1_76d3ad45"

TREE_FEATURES = [
    "users_on_device_id_72h", "users_on_device_id_7d",
    "users_on_persistent_account_id_72h", "users_on_persistent_account_id_7d",
    "users_on_phone_72h", "users_on_phone_7d",
    "users_on_email_72h", "users_on_email_7d",
    "users_on_address_72h", "users_on_address_7d",
    "users_on_bank_account_7d",
    "hours_since_identity_created", "days_since_plaid_account_created",
    "days_between_identity_and_bank_account_creation",
    "prior_min_hours_between_advances_on_account", "avg_prior_advances_per_day",
    "loan_amount", "total_disbursed", "is_neobank_high_risk_institution",
    "name_match_last", "name_match_first", "bank_accounts_per_user_asof", "has_kyc",
    # extra discriminators to split the broad neobank/small/fresh cohort finer
    "hours_since_socure_created", "origination_hour", "signup_ip_matches_latest_ip",
    "express_transfer_fee", "prior_advances_on_bank_account_lifetime",
]


def _load_env() -> None:
    p = Path(".env")
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--min-leaf", type=int, default=300, help="smaller = narrower, purer pockets")
    ap.add_argument("--min-test", type=int, default=20, help="min held-out rows to report a leaf")
    ap.add_argument("--sort", choices=("never", "dpd45"), default="dpd45")
    args = ap.parse_args()

    _load_env()
    from sklearn.tree import DecisionTreeClassifier
    from automl.data.registry import load_dataset_by_id
    from automl.project.session import use_project
    from projects.fraud_anomaly_detection.scenarios import SCENARIOS, residual_mask

    sess = use_project("fraud_anomaly_detection", dry_run=False)
    df = load_dataset_by_id(DATASET_ID, session=sess).df

    print(f"registered scenarios gated out: {[s.name for s in SCENARIOS]}")
    res = residual_mask(df).to_numpy()
    mat = (df["label_mature_d45"].astype(float) == 1).to_numpy()
    dpd = (df["label_gross_dpd45"].astype(float) == 1).to_numpy()
    never = dpd & (df["label_repaid_current_snapshot"].astype(float) == 0).to_numpy()
    split = pd.to_numeric(df["SPLIT_PCT"], errors="coerce").to_numpy(float)

    keep = res & mat
    sub = df[keep].reset_index(drop=True)
    dpd_k, never_k = dpd[keep], never[keep]
    split_k = split[keep]
    print(f"post-lock residual + mature: {keep.sum():,} rows | "
          f"DPD45 base {dpd_k.mean():.3%} | never-paid base {never_k.mean():.3%}")

    feats = [c for c in TREE_FEATURES if c in sub.columns]
    X = sub[feats].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median()).to_numpy(float)
    tr, te = split_k < 80, split_k >= 80
    base_dpd_te, base_never_te = dpd_k[te].mean(), never_k[te].mean()
    print(f"features {len(feats)} | train {tr.sum():,} | test {te.sum():,} | "
          f"test DPD45 base {base_dpd_te:.3%}\n")

    tree = DecisionTreeClassifier(max_depth=args.max_depth, min_samples_leaf=args.min_leaf,
                                  class_weight="balanced", random_state=0)
    tree.fit(X[tr], dpd_k[tr].astype(int))   # TARGET = DPD45 (matured only)

    t = tree.tree_
    leaves: list[dict] = []

    def walk(node, conds, m_tr, m_te):
        if t.children_left[node] == -1:
            leaves.append({"conds": list(conds), "tr": m_tr, "te": m_te})
            return
        f, thr = feats[t.feature[node]], t.threshold[node]
        col = X[:, feats.index(f)]
        l_tr, l_te = col[tr] <= thr, col[te] <= thr
        walk(t.children_left[node], conds + [f"{f} <= {thr:.4g}"], m_tr & l_tr, m_te & l_te)
        walk(t.children_right[node], conds + [f"{f} > {thr:.4g}"], m_tr & ~l_tr, m_te & ~l_te)

    walk(0, [], np.ones(tr.sum(), bool), np.ones(te.sum(), bool))

    rows = []
    for lf in leaves:
        n_te = int(lf["te"].sum())
        if n_te < args.min_test:
            continue
        d_te = dpd_k[te][lf["te"]].mean()
        nv_te = never_k[te][lf["te"]].mean()
        rows.append({
            "conds": " AND ".join(lf["conds"]), "n_te": n_te,
            "dpd45": d_te, "dpd_lift": d_te / base_dpd_te,
            "never": nv_te, "never_lift": nv_te / base_never_te,
            # fraud smell: of the DPD45 in this leaf, how many never repaid
            "frac_never_of_dpd": (nv_te / d_te) if d_te else float("nan"),
        })
    rows.sort(key=lambda r: r["never" if args.sort == "never" else "dpd45"], reverse=True)
    print(f"top leaves by held-out test {args.sort} precision "
          f"(min {args.min_test} test rows; depth {args.max_depth}, min_leaf {args.min_leaf}):")
    print("  (frac_never_of_dpd near 1.0 = fraud-shaped; near base ~0.87 = credit stress)\n")
    for r in rows[:15]:
        print(f"  • {r['conds']}")
        print(f"      n_te={r['n_te']}  DPD45={r['dpd45']:.1%} ({r['dpd_lift']:.1f}x)  "
              f"never-paid={r['never']:.1%} ({r['never_lift']:.1f}x)  "
              f"fraud-smell={r['frac_never_of_dpd']:.0%} of DPD45 never repaid")


if __name__ == "__main__":
    main()
