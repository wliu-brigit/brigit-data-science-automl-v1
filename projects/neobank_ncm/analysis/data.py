"""Loaders for the downstream-analysis inputs.

Three frozen sandbox snapshots plus the one live LTV pull (see the SQL files
under data/queries/analysis/ — each cites its legacy source). Loaders use
the harness Snowflake helper and lowercase column names on the way in (this
repo's data convention; the legacy worked in UPPER-no-underscore).

Every loader takes ``frame=`` / ``parquet=`` overrides so tests and cached
runs never touch the warehouse; without them, offline use fails with a
clear VPN message.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
QUERIES_DIR = PROJECT_DIR / "data" / "queries" / "analysis"
LEGACY_DIR = PROJECT_DIR / "data" / "legacy"

# derived/encoded model inputs that do NOT exist in the daily table — they
# are computed at scoring time (legacy financial notebook cell 6)
_NOT_IN_TABLE = {
    "BALANCESDTODAILYINCOMEMEANRATIO",
    "MAXNEGBALANCE30DTODAILYINCOMEMEANRATIO",
    "INFLOWSUMTOOUTFLOWSUMRATIO14D",
    "NETFLOWTODAILYINCOMEMEANRATIO14D",
    "BALANCEDEPLETIONRATE1D",
    "INCOMEBUFFERTODAYSTOPAYDAYRATIO",
    "COMPETITORBORROWINTENSITY",
    "ISTAXSEASON",
    "BANKINSTITUTIONWOE",
}

# metadata, outcomes, and derived-feature inputs beyond the locked feature
# list (legacy cell 6 ``extra_cols``)
_EXTRA_COLS = [
    "user_id",
    "first_report_date",
    "snapshot_date",
    "day_number",
    "went_dpd45",
    "synthetic_score",
    "v2_score",
    "loan_amount_max",
    "bankinstitution",
    "daystopayday",
    "earninsummarycreditninetydayamount",
    "highestpaydepositmean",
    "maxnegativebalpast30days",
    "outflowsum14d",
    "account_approval_state",
    "plaidfeaturessummary_incomewages_lookbackwindow30d_inflow_sum",
]


def _sql(name: str) -> str:
    return (QUERIES_DIR / name).read_text(encoding="utf-8")


def _fetch(sql: str) -> pd.DataFrame:
    from automl.utils.io import snowflake as sf

    missing = sf.missing_env()
    if missing:
        raise EnvironmentError(
            f"Snowflake not configured ({', '.join(missing)} unset) — this loader "
            "needs the VPN; offline, pass frame=/parquet= instead"
        )
    return sf.fetch_df(sql)


def _resolve(frame: pd.DataFrame | None, parquet: str | Path | None) -> pd.DataFrame | None:
    if frame is not None:
        return frame.copy()
    if parquet is not None:
        return pd.read_parquet(parquet)
    return None


def daily_select_columns() -> list[str]:
    """The legacy cell-6 column list: extras + locked features, deduped.

    The locked feature list keeps the experiment-phase Snowflake names
    (snake_case) — the reverse map resolves the normalized names back to
    the physical columns, exactly as the legacy did.
    """
    features_meta = json.loads((LEGACY_DIR / "features.json").read_text())
    feature_cols = (
        features_meta["NUMERICAL_FEATURES"]
        + features_meta["CATEGORICAL_FEATURES"]
        + features_meta["BOOL_FEATURES"]
    )
    decisions = json.loads((LEGACY_DIR / "experiment_decisions.json").read_text())
    orig_map = {c.upper().replace("_", ""): c for c in decisions["feature_cols"]}
    table_feature_cols = [
        orig_map.get(c, c.lower()) for c in feature_cols if c not in _NOT_IN_TABLE
    ]
    return list(dict.fromkeys(_EXTRA_COLS + table_feature_cols))


def load_daily(
    *,
    frame: pd.DataFrame | None = None,
    parquet: str | Path | None = None,
    columns: list[str] | str | None = None,
) -> pd.DataFrame:
    """The D1–D30 daily snapshot, lowercase, with is_known + parsed dates."""
    daily = _resolve(frame, parquet)
    if daily is None:
        if columns is None:
            columns = daily_select_columns()
        cols_sql = columns if isinstance(columns, str) else ",\n    ".join(columns)
        daily = _fetch(_sql("oot_new_links_daily.sql").format(columns=cols_sql))
    daily.columns = [c.lower() for c in daily.columns]
    daily["is_known"] = daily["went_dpd45"].notna()
    for col in ("snapshot_date", "first_report_date"):
        if col in daily.columns:
            daily[col] = pd.to_datetime(daily[col])
    return daily


def load_ri_scores(
    *, frame: pd.DataFrame | None = None, parquet: str | Path | None = None
) -> pd.DataFrame:
    """The frozen new-links RI scores (consumed as-is; RI model never re-run)."""
    scores = _resolve(frame, parquet)
    if scores is None:
        scores = _fetch(_sql("oot_new_links_ri_scores.sql"))
    scores.columns = [c.lower() for c in scores.columns]
    return scores


def load_synthetic_scores_oot(
    *, frame: pd.DataFrame | None = None, parquet: str | Path | None = None
) -> pd.DataFrame:
    """The Phase-4 synthetic scores, oot split (for the RI-consistency QA)."""
    scores = _resolve(frame, parquet)
    if scores is None:
        scores = _fetch(_sql("synthetic_scores_oot.sql"))
    scores.columns = [c.lower() for c in scores.columns]
    return scores


def load_user_ltv(
    *, frame: pd.DataFrame | None = None, parquet: str | Path | None = None
) -> pd.DataFrame:
    """Per-user revenue/LTV aggregation — THE one live (non-snapshot) pull."""
    ltv = _resolve(frame, parquet)
    if ltv is None:
        ltv = _fetch(_sql("user_ltv.sql"))
    ltv.columns = [c.lower() for c in ltv.columns]
    ltv["is_activated"] = ltv["first_activation_date"].notna()
    return ltv


def ri_scores_parity(
    ri_scores: pd.DataFrame, synthetic_scores_oot: pd.DataFrame
) -> dict[str, float]:
    """RI-consistency QA (legacy oot_new_links_ri_scoring score-comparison cell).

    Inner-joins the new-links RI scores against the Phase-4 oot scores and
    reports agreement; the legacy run saw corr 0.9999 and 99% exact
    percentile agreement.
    """
    cmp_df = ri_scores.merge(synthetic_scores_oot, on="user_id", how="inner")
    new = cmp_df["synthetic_score"].astype(float)
    old = cmp_df["final_score"].astype(float)
    diff = (new - old).abs()
    pct_new = pd.qcut(new, 100, labels=False, duplicates="drop")
    pct_old = pd.qcut(old, 100, labels=False, duplicates="drop")
    return {
        "n_overlap": int(len(cmp_df)),
        "pearson_corr": float(new.corr(old)),
        "mean_abs_diff": float(diff.mean()),
        "median_abs_diff": float(diff.median()),
        "p90_abs_diff": float(diff.quantile(0.90)),
        "p99_abs_diff": float(diff.quantile(0.99)),
        "mean_signed_diff": float((new - old).mean()),
        "pct_exact_percentile": float((pct_new == pct_old).mean()),
        "pct_within_1_percentile": float((np.abs(pct_new - pct_old) <= 1).mean()),
    }
