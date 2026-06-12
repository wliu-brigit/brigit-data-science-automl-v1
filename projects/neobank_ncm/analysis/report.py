"""Assemble the full decision report (all scenarios, settled vocabulary).

Pure function over a *scored* daily frame (``v3_score`` present) that also
carries the per-user LTV columns (broadcast). Reuses the policy/impact helpers;
this module owns the legacy->settled rename and the all-scenario sweep. See
docs/to-do/decision-metric-vocabulary.md for the contract.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from projects.neobank_ncm.analysis import impact, policy, scoring

_TRACK_RENAME = {
    "v3_thr": "candidate_score_cutoff",
    "n_v3": "candidate_approved_count",
    "v3_ar": "candidate_approval_rate",
    "ref_ar": "v3a_approval_rate",
    "delta_ar": "approval_rate_delta",
    "v3_br": "candidate_bad_rate",
    "ref_br": "v3a_bad_rate",
    "delta_br": "bad_rate_delta",
    "swap_in_br": "swap_in_bad_rate",
    "swap_out_br": "swap_out_bad_rate",
    "swap_in_vol": "swap_in_count",
    "swap_out_vol": "swap_out_count",
    "d1_ar": "day1_approval_rate",
}

# (scenario_key, ko_gate, objective, table_key)
# table_key matches all_threshold_tables output: "<variant>/match_<ar|br>"
_SCENARIOS = {
    1: ("1_no_ko_match_bad_rate",          "none",            "match_bad_rate",     "no_ko/match_br"),
    2: ("2_income500_match_bad_rate",       "income500",       "match_bad_rate",     "income500/match_br"),
    3: ("3_v3a_ko_match_bad_rate",          "v3a_ko",          "match_bad_rate",     "v3a_ko/match_br"),
    7: ("7_income500_broad_match_bad_rate", "income500_broad", "match_bad_rate",     "income500_broad/match_br"),
    4: ("4_no_ko_match_approval_rate",      "none",            "match_approval_rate","no_ko/match_ar"),
    5: ("5_income500_match_approval_rate",  "income500",       "match_approval_rate","income500/match_ar"),
    6: ("6_v3a_ko_match_approval_rate",     "v3a_ko",          "match_approval_rate","v3a_ko/match_ar"),
}

# Row labels emitted by policy.threshold_table — must match exactly.
# Note the double-space after "UW" (verified against policy.py line 347).
_UW_LABEL = "UW  (v2<=0.485 + KOs)"
_CLE_LABEL = "CLE (v2<=0.64 + inc>500)"


def _scalar(value):
    """Coerce numpy scalar to a plain Python number."""
    if isinstance(value, np.integer):
        return int(value)
    return float(value)


def _rename_track(row: pd.Series) -> dict:
    """Map one table row (legacy column names) to the settled vocabulary."""
    return {settled: _scalar(row[legacy]) for legacy, settled in _TRACK_RENAME.items()}


def _user_ltv_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Extract per-user LTV columns from the broadcast daily frame.

    The daily frame carries LTV columns broadcast from the user-LTV snapshot
    (one value per user repeated across all their rows). We deduplicate to a
    user-grain frame and derive ``is_activated`` from ``first_activation_date``.
    """
    cols = (
        ["user_id", "loan_amount_max", "underwriting_strategy", "first_activation_date"]
        + [f"total_revenue_{h}" for h in impact.HORIZONS]
        + [f"total_ltv_lite_{h}" for h in impact.HORIZONS]
        + [f"ltv_{h}_elig" for h in impact.HORIZONS]
    )
    ltv = daily[cols].drop_duplicates("user_id").reset_index(drop=True)
    ltv["is_activated"] = ltv["first_activation_date"].notna()
    return ltv


def _scenario_ltv(
    daily: pd.DataFrame,
    users: pd.DataFrame,
    thresholds: dict,
    scenario_id: int,
) -> dict:
    """LTV-per-link at D90 and D120 for one scenario."""
    us_raw = impact.merge_ltv(users, _user_ltv_from_daily(daily))
    ref = impact.historical_reference(us_raw)
    us, lkp = impact.build_lookup(us_raw)

    _, thr_uw, thr_cle, ko_uw, ko_cle = policy.scenario_map(thresholds)[scenario_id]
    arr = policy.first_approval_days(daily, users["user_id"], thr_uw, thr_cle, ko_uw, ko_cle)

    uw_mask = us["user_id"].isin(set(users["user_id"][arr["uw"] <= 30]))
    cle_mask = (
        us["user_id"].isin(set(users["user_id"][arr["cle"] <= 30]))
        & ~uw_mask
    )
    lam = pd.Series(
        np.where(uw_mask, policy.LAM_UW, np.where(cle_mask, policy.LAM_CLE, np.nan)),
        index=us.index,
    )
    approved_mask = uw_mask | cle_mask
    agg = impact.monthly_aggregate(
        impact.infer_financials(us, lkp, approved_mask, lam),
        len(us),
        ref["act_rate"],
        ref["monthly_vol"],
    )
    return {
        "ltv_per_link_d90": round(float(agg["lpl_90"]), 4),
        "ltv_per_link_d120": round(float(agg["lpl_120"]), 4),
    }


def build_decision_report(
    daily: pd.DataFrame,
    *,
    headline_scenario: int = 2,
    provenance: dict | None = None,
) -> dict:
    """Assemble the full decision report over all 7 scenarios.

    Parameters
    ----------
    daily:
        Scored daily frame — must already carry ``v3_score`` plus the LTV
        columns (``total_revenue_{h}``, ``total_ltv_lite_{h}``, ``ltv_{h}_elig``
        for h in 30/60/90/120, and ``loan_amount_max``, ``underwriting_strategy``,
        ``first_activation_date``).
    headline_scenario:
        Which scenario id (1–7) to record in provenance as the headline.
        Defaults to 2 (income>500 BR-match, the legacy PICKED scenario).
    provenance:
        Optional caller-supplied provenance dict (run_id, dataset_id, etc.).
        If provided it is included under ``"provenance"`` with ``headline_scenario``
        appended.

    Returns
    -------
    dict
        Structured report per docs/to-do/decision-metric-vocabulary.md.
    """
    policy.add_policy_columns(daily)
    users = policy.collapse_to_users(daily)
    bench = policy.benchmarks(users)
    thresholds = policy.compute_thresholds(users, bench)
    tables = policy.all_threshold_tables(users, thresholds)
    auc = scoring.d2_known_auc(daily)

    scenarios: dict = {}
    for scenario_id, (key, ko_gate, objective, table_key) in _SCENARIOS.items():
        table = tables[table_key]
        ltv = _scenario_ltv(daily, users, thresholds, scenario_id)
        scenarios[key] = {
            "ko_gate": ko_gate,
            "objective": objective,
            **ltv,
            "tracks": {
                "uw": _rename_track(table.loc[_UW_LABEL]),
                "cle": _rename_track(table.loc[_CLE_LABEL]),
            },
        }

    out: dict = {
        "discrimination": {
            "day2_known_auc": round(float(auc["d2_auc"]), 5),
            "day2_known_count": int(auc["d2_n"]),
            "day2_known_bad_rate": round(float(auc["d2_bad_rate"]), 5),
        },
        "benchmark": {
            "v3a_approval_rate": round(float(bench["v3a_ar"]), 5),
            "v3a_bad_rate": round(float(bench["v3a_br"]), 5),
            "cle_approval_rate": round(float(bench["cle_ar"]), 5),
            "cle_bad_rate": round(float(bench["cle_br"]), 5),
        },
        "scenarios": scenarios,
    }
    if provenance is not None:
        out["provenance"] = {
            **provenance,
            "headline_scenario": _SCENARIOS[headline_scenario][0],
        }
    return out
