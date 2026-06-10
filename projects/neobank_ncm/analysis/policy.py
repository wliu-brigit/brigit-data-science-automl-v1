"""Approval-policy logic: KO rules, user collapse, thresholds, scenarios.

Faithful port of cells 12–16, 21 and 26 of the legacy
financial_impact_analysis.ipynb. Column names are lowercase (this repo's
data convention); every constant cites the legacy cell it came from.

Semantics worth knowing (legacy cell 12):

- A user's policy score is the MIN of the daily model score over the days
  on which the policy's gates pass — the score is set to NaN on ineligible
  days so those days can never contribute.
- ``effective_bad`` is the real outcome for known users and the synthetic
  score for unknowns. It exists for policy analysis only; it must never
  feed training or the leaderboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Locked policy constants (legacy cells 12, 14, 21) ───────────────────────
V2_CUTOFF_UW = 0.485   # v3a underwriting track: v2_score <= 0.485
V2_CUTOFF_CLE = 0.64   # CLE track: v2_score <= 0.64
KO_INCOME_UW = 700.0   # v3a KO: monthly income > 700
KO_PAYCHECK_UW = 300.0  # v3a KO: paycheck > 300
KO_NOACTIVITY_UW = 0.8  # v3a KO: noactivityrate < 0.8
KO_INCOME_500 = 500.0  # income>500 KO variants
LAM_UW = 50.0          # loan amount ($) for UW-track approvals
LAM_CLE = 25.0         # loan amount ($) for CLE-only approvals
APPROVAL_WINDOW_DAYS = 30  # any-day approval = first pass within D1–D30

PLAID_INFLOW_30D = "plaidfeaturessummary_incomewages_lookbackwindow30d_inflow_sum"


# ── Daily policy columns (legacy cell 12) ────────────────────────────────────
def add_policy_columns(daily: pd.DataFrame) -> pd.DataFrame:
    """Add the per-day eligibility / KO / incumbent-policy columns in place."""
    daily["monthly_income_daily"] = daily["dailyincomemean"].abs() * 30
    daily["paycheck_daily"] = daily["highestpaydepositmean"].abs()

    # account gate: NOT_APPROVED days are ineligible for every scenario;
    # NULL state counts as eligible
    daily["acct_eligible"] = daily["account_approval_state"].isna() | daily[
        "account_approval_state"
    ].ne("NOT_APPROVED")

    daily["ko_pass_daily"] = (
        (daily["monthly_income_daily"] > KO_INCOME_UW)
        & (daily["paycheck_daily"] > KO_PAYCHECK_UW)
        & (daily["noactivityrate"].astype(float) < KO_NOACTIVITY_UW)
    )
    daily["ko_pass_daily_500"] = daily["monthly_income_daily"] > KO_INCOME_500
    daily["ko_pass_daily_500_broad"] = (daily["monthly_income_daily"] > KO_INCOME_500) | (
        daily[PLAID_INFLOW_30D].abs() > KO_INCOME_500
    )

    daily["v3a_pass_daily"] = (
        (daily["v2_score"].astype(float) <= V2_CUTOFF_UW)
        & daily["ko_pass_daily"]
        & daily["acct_eligible"]
    )
    daily["cle_pass_daily"] = (
        (daily["v2_score"].astype(float) <= V2_CUTOFF_CLE)
        & (daily["monthly_income_daily"] > KO_INCOME_500)
        & daily["acct_eligible"]
    )
    return daily


# ── User-level collapse (legacy cell 12) ─────────────────────────────────────
def collapse_to_users(daily: pd.DataFrame) -> pd.DataFrame:
    """One row per user: min eligible-day scores, approvals, effective_bad.

    Requires ``v3_score`` (from scoring.score_daily) and the policy columns
    (from add_policy_columns). Rows with NULL effective_bad are dropped, as
    in the legacy.
    """
    user_scores = (
        daily.assign(
            # score is NaN on ineligible days so it cannot contribute to the min
            v3_score_elig=lambda d: np.where(d["acct_eligible"], d["v3_score"], np.nan),
            v3_score_ko=lambda d: np.where(
                d["ko_pass_daily"] & d["acct_eligible"], d["v3_score"], np.nan
            ),
            v3_score_ko500=lambda d: np.where(
                d["ko_pass_daily_500"] & d["acct_eligible"], d["v3_score"], np.nan
            ),
            v3_score_ko500_broad=lambda d: np.where(
                d["ko_pass_daily_500_broad"] & d["acct_eligible"], d["v3_score"], np.nan
            ),
        )
        .groupby("user_id")
        .agg(
            is_known=("is_known", "first"),
            went_dpd45=("went_dpd45", "first"),
            synthetic_score=("synthetic_score", "first"),
            min_v3_score=("v3_score_elig", "min"),
            min_v3_score_ko=("v3_score_ko", "min"),
            min_v3_score_ko500=("v3_score_ko500", "min"),
            min_v3_score_ko500_broad=("v3_score_ko500_broad", "min"),
            v3a_approved=("v3a_pass_daily", "any"),
            days_with_data=("day_number", "count"),
            days_ko_pass=("ko_pass_daily", "sum"),
            days_ko_pass_500=("ko_pass_daily_500", "sum"),
            days_ko_pass_500_broad=("ko_pass_daily_500_broad", "sum"),
            cle_approved=("cle_pass_daily", "any"),
        )
        .reset_index()
    )

    user_scores["effective_bad"] = (
        user_scores["went_dpd45"].fillna(user_scores["synthetic_score"]).astype(float)
    )
    user_scores = user_scores.dropna(subset=["effective_bad"]).reset_index(drop=True)

    # D1 snapshot for D1 approval-rate reporting. The legacy comment said to
    # null the score on acct-ineligible D1 rows, but its executed code never
    # did: the assign wrote a dead lowercase 'v3_score' column while the
    # selection took the raw uppercase 'V3_SCORE'. We replicate the EXECUTED
    # behavior (raw score) for exact parity; the D1-AR diagnostic therefore
    # counts acct-ineligible day-1 rows, as the legacy numbers do.
    d1_rows = (
        daily[daily["day_number"] == 1][
            [
                "user_id",
                "v3_score",
                "ko_pass_daily",
                "v3a_pass_daily",
                "ko_pass_daily_500",
                "ko_pass_daily_500_broad",
                "cle_pass_daily",
            ]
        ].rename(
            columns={
                "v3_score": "d1_v3_score",
                "ko_pass_daily": "d1_ko_pass",
                "v3a_pass_daily": "d1_v3a_pass",
                "ko_pass_daily_500": "d1_ko_pass_500",
                "ko_pass_daily_500_broad": "d1_ko_pass_500_broad",
                "cle_pass_daily": "d1_cle_pass",
            }
        )
    )
    return user_scores.merge(d1_rows, on="user_id", how="left")


def eligibility_masks(user_scores: pd.DataFrame) -> dict[str, pd.Series]:
    """The standing masks of legacy cell 12 (V3A/CLE/KO-eligible)."""
    return {
        "v3a": user_scores["v3a_approved"],
        "cle": user_scores["cle_approved"],
        "ko": user_scores["min_v3_score_ko"].notna(),
        "ko500": user_scores["min_v3_score_ko500"].notna(),
        "ko500_broad": user_scores["min_v3_score_ko500_broad"].notna(),
    }


# ── Benchmarks (legacy cell 13) ──────────────────────────────────────────────
def known_br(mask: pd.Series, df: pd.DataFrame) -> float:
    """Bad rate on ground truth only: known users inside ``mask``."""
    known = mask & df["is_known"]
    n_known = int(known.sum())
    if not n_known:
        return float("nan")
    return float(df.loc[known, "went_dpd45"].astype(float).sum()) / n_known


def benchmarks(user_scores: pd.DataFrame) -> dict[str, float]:
    """v3a and CLE incumbent metrics; bad rates are known-only ground truth."""
    n = len(user_scores)
    masks = eligibility_masks(user_scores)
    out: dict[str, float] = {"n": n}
    for name in ("v3a", "cle"):
        mask = masks[name]
        out[f"n_{name}"] = int(mask.sum())
        out[f"{name}_ar"] = out[f"n_{name}"] / n
        br = known_br(mask, user_scores)
        # legacy cell 13 falls back to 0.0 (not NaN) with no known approvals
        out[f"{name}_br"] = 0.0 if np.isnan(br) else br
        d1 = user_scores[f"d1_{name}_pass"].fillna(False)
        out[f"{name}_d1_ar"] = float(d1.sum()) / n
    return out


# ── Threshold search (legacy cells 14–16, unified) ───────────────────────────
def threshold_for_ar(
    eligible_scores: np.ndarray, target_ar: float, n_total: int
) -> float:
    """Score cutoff whose approval count over n_total matches target_ar.

    ``eligible_scores`` are the min-scores of the policy-eligible users
    (NaN-free); approval rate is counted against the full population.
    """
    valid = np.asarray(eligible_scores, dtype=float)
    valid = valid[~np.isnan(valid)]
    n_target = target_ar * n_total
    if n_target <= 0 or not len(valid):
        return 0.0
    frac = min(n_target / len(valid), 1.0)
    return float(np.quantile(valid, frac))


def threshold_for_br(
    user_scores: pd.DataFrame, target_br: float, score_col: str
) -> float:
    """Score cutoff whose cumulative effective bad rate matches target_br."""
    sdf = user_scores[["effective_bad", score_col]].dropna()
    if sdf.empty:  # nobody passes this KO gate -> approve nobody
        return 0.0
    sdf = sdf.sort_values(score_col).reset_index(drop=True)
    n = len(sdf)
    cum_br = sdf["effective_bad"].cumsum().values / np.arange(1, n + 1, dtype=float)
    idx = int(np.argmin(np.abs(cum_br - target_br)))
    return float(sdf[score_col].iloc[idx])


def compute_thresholds(user_scores: pd.DataFrame, bench: dict[str, float]) -> dict[str, float]:
    """All twelve legacy thresholds (cells 14–16), keyed t{1,2}{variant}_{uw,cle}.

    1 = match approval rate, 2 = match bad rate; variants: '' (no KO),
    k5 (income>500), k5b (income>500 broad), k (v3a KOs; CLE row uses the
    income>500 pool, as in legacy cell 16).
    """
    n = bench["n"]
    score = {
        "": "min_v3_score",
        "k5": "min_v3_score_ko500",
        "k5b": "min_v3_score_ko500_broad",
        "k": "min_v3_score_ko",
    }
    out: dict[str, float] = {}
    for variant, col in score.items():
        for ref in ("uw", "cle"):
            target_ar = bench["v3a_ar"] if ref == "uw" else bench["cle_ar"]
            target_br = bench["v3a_br"] if ref == "uw" else bench["cle_br"]
            # legacy cell 16: the CLE row of the v3a-KO table uses the
            # income>500 pool, not the v3a-KO pool
            ref_col = "min_v3_score_ko500" if (variant == "k" and ref == "cle") else col
            out[f"t1{variant}_{ref}"] = threshold_for_ar(
                user_scores[ref_col].values, target_ar, n
            )
            out[f"t2{variant}_{ref}"] = threshold_for_br(user_scores, target_br, ref_col)
    return out


# ── Comparison metrics + tables (legacy cells 14–16) ─────────────────────────
def extended_metrics(
    ref_mask: pd.Series, v3_mask: pd.Series, user_scores: pd.DataFrame
) -> dict[str, float]:
    """Reference-vs-proposed comparison incl. swap-in/swap-out populations."""
    n = len(user_scores)
    n_ref = int(ref_mask.sum())
    n_v3 = int(v3_mask.sum())
    v3_br = (
        float(user_scores.loc[v3_mask, "effective_bad"].sum()) / n_v3 if n_v3 else 0.0
    )
    ref_br = known_br(ref_mask, user_scores)
    si_mask = v3_mask & ~ref_mask
    so_mask = ref_mask & ~v3_mask
    n_in = int(si_mask.sum())
    n_out = int(so_mask.sum())
    in_br = (
        float(user_scores.loc[si_mask, "effective_bad"].sum()) / n_in
        if n_in
        else float("nan")
    )
    out_br = (
        float(user_scores.loc[so_mask, "effective_bad"].sum()) / n_out
        if n_out
        else float("nan")
    )
    return dict(
        n_v3=n_v3,
        v3_ar=n_v3 / n,
        ref_ar=n_ref / n,
        delta_ar=n_v3 / n - n_ref / n,
        v3_br=v3_br,
        ref_br=ref_br,
        delta_br=v3_br - ref_br,
        swap_in_br=in_br,
        swap_out_br=out_br,
        swap_in_vol=n_in,
        swap_out_vol=n_out,
    )


# table layout shared by every threshold table (legacy cell 14)
TABLE_COLUMNS = [
    "v3_thr",
    "n_v3",
    "v3_ar",
    "ref_ar",
    "delta_ar",
    "v3_br",
    "ref_br",
    "delta_br",
    "swap_in_br",
    "swap_out_br",
    "swap_in_vol",
    "swap_out_vol",
    "d1_ar",
]

# variant -> (score_col, d1 KO column or None) for proposal masks
_VARIANTS = {
    "": ("min_v3_score", None),
    "k5": ("min_v3_score_ko500", "d1_ko_pass_500"),
    "k5b": ("min_v3_score_ko500_broad", "d1_ko_pass_500_broad"),
    "k": ("min_v3_score_ko", "d1_ko_pass"),
}


def proposal_masks(
    user_scores: pd.DataFrame, threshold: float, variant: str, *, cle_row: bool = False
) -> tuple[pd.Series, pd.Series]:
    """(any-day approved, d1 approved) for one table row (legacy cells 14–16)."""
    if variant == "k" and cle_row:
        variant = "k5"  # legacy cell 16: CLE row swaps to the income>500 pool
    score_col, d1_ko_col = _VARIANTS[variant]
    if variant == "":
        approved = user_scores[score_col] <= threshold
        d1 = user_scores["d1_v3_score"].fillna(1.0) <= threshold
    else:
        approved = user_scores[score_col].notna() & (user_scores[score_col] <= threshold)
        d1 = user_scores[d1_ko_col].fillna(False) & (
            user_scores["d1_v3_score"].fillna(1.0) <= threshold
        )
    return approved, d1


def threshold_table(
    user_scores: pd.DataFrame,
    thresholds: dict[str, float],
    variant: str,
    objective: str,
) -> pd.DataFrame:
    """One legacy table: UW + CLE rows for a KO variant and matching objective.

    ``objective`` is 'ar' (t1*) or 'br' (t2*); ``variant`` is '', 'k5',
    'k5b' or 'k'. Values are raw floats — format at display time.
    """
    masks = eligibility_masks(user_scores)
    prefix = "t1" if objective == "ar" else "t2"
    rows = {}
    for ref, ref_mask, label in (
        ("uw", masks["v3a"], "UW  (v2<=0.485 + KOs)"),
        ("cle", masks["cle"], "CLE (v2<=0.64 + inc>500)"),
    ):
        t = thresholds[f"{prefix}{variant}_{ref}"]
        approved, d1 = proposal_masks(user_scores, t, variant, cle_row=(ref == "cle"))
        m = extended_metrics(ref_mask, approved, user_scores)
        rows[label] = {"v3_thr": t, **m, "d1_ar": float(d1.mean())}
    return pd.DataFrame.from_dict(rows, orient="index")[TABLE_COLUMNS]


def all_threshold_tables(
    user_scores: pd.DataFrame, thresholds: dict[str, float]
) -> dict[str, pd.DataFrame]:
    """Every legacy threshold table, keyed '<variant-label>/<objective>'."""
    labels = {"": "no_ko", "k5": "income500", "k5b": "income500_broad", "k": "v3a_ko"}
    return {
        f"{label}/match_{objective}": threshold_table(
            user_scores, thresholds, variant, objective
        )
        for variant, label in labels.items()
        for objective in ("ar", "br")
    }


# ── Scenarios + first-approval days (legacy cells 21, 26) ────────────────────
def scenario_map(thresholds: dict[str, float]) -> dict[int, tuple]:
    """The legacy seven scenarios: (label, thr_uw, thr_cle, ko_col_uw, ko_col_cle)."""
    t = thresholds
    return {
        1: ("No-KO  BR-match", t["t2_uw"], t["t2_cle"], None, None),
        2: ("income>500 BR-match", t["t2k5_uw"], t["t2k5_cle"], "ko_pass_daily_500", "ko_pass_daily_500"),
        3: ("V3A KOs BR-match", t["t2k_uw"], t["t2k_cle"], "ko_pass_daily", "ko_pass_daily_500"),
        4: ("No-KO  AR-match", t["t1_uw"], t["t1_cle"], None, None),
        5: ("income>500 AR-match", t["t1k5_uw"], t["t1k5_cle"], "ko_pass_daily_500", "ko_pass_daily_500"),
        6: ("V3A KOs AR-match", t["t1k_uw"], t["t1k_cle"], "ko_pass_daily", "ko_pass_daily_500"),
        7: ("income>500 broad BR-match", t["t2k5b_uw"], t["t2k5b_cle"], "ko_pass_daily_500_broad", "ko_pass_daily_500_broad"),
    }


def first_approval_days(
    daily: pd.DataFrame,
    user_ids: pd.Series,
    thr_uw: float,
    thr_cle: float,
    ko_col_uw: str | None,
    ko_col_cle: str | None,
) -> dict[str, np.ndarray]:
    """First passing day per user for UW / CLE / combined / v3a (legacy cell 21).

    Arrays align with ``user_ids`` (the user_scores order); NaN = never.
    """

    def _track(threshold: float, ko_col: str | None) -> pd.Series:
        passing = daily["acct_eligible"] & (daily["v3_score"] <= threshold)
        if ko_col:
            passing = daily[ko_col] & passing
        return daily[passing].groupby("user_id")["day_number"].min()

    uw_arr = _track(thr_uw, ko_col_uw).reindex(user_ids).values.astype(float)
    cle_arr = _track(thr_cle, ko_col_cle).reindex(user_ids).values.astype(float)
    v3a_first = daily[daily["v3a_pass_daily"]].groupby("user_id")["day_number"].min()
    return {
        "uw": uw_arr,
        "cle": cle_arr,
        "v3": np.fmin(uw_arr, cle_arr),
        "v3a": v3a_first.reindex(user_ids).values.astype(float),
    }


def approval_curves(
    arrays: dict[str, np.ndarray], days: np.ndarray | None = None
) -> pd.DataFrame:
    """Cumulative approval-rate curves (legacy cell 26: D2–D30)."""
    if days is None:
        days = np.arange(2, APPROVAL_WINDOW_DAYS + 1)
    return pd.DataFrame(
        {name: [(arr <= d).mean() for d in days] for name, arr in arrays.items()},
        index=pd.Index(days, name="day"),
    )
