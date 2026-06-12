# projects/neobank_ncm/tests/test_decision_report.py
from __future__ import annotations

import numpy as np
import pandas as pd

from projects.neobank_ncm.analysis import policy, report, scoring
from projects.neobank_ncm.tests.fixtures import make_synthetic_daily


def _scored_daily_with_ltv():
    daily = make_synthetic_daily()                 # synthetic D1-D30 daily frame
    daily["v3_score"] = np.clip(                   # stand-in candidate score
        daily["v2_score"].astype(float) + np.random.default_rng(0).normal(0, 0.05, len(daily)),
        0, 1,
    )
    # broadcast cheap synthetic LTV columns per user (as the real frame will carry)
    for h in (30, 60, 90, 120):
        daily[f"total_revenue_{h}"] = 2.0
        daily[f"total_ltv_lite_{h}"] = 1.5
        daily[f"ltv_{h}_elig"] = True
    daily["loan_amount_max"] = 50.0
    daily["underwriting_strategy"] = "UNDERWRITING_NEOBANK_STRATEGY_V3A"
    daily["first_activation_date"] = pd.Timestamp("2026-01-15")
    return daily


def test_decision_report_structure_and_naming():
    rep = report.build_decision_report(_scored_daily_with_ltv(), headline_scenario=2)

    assert set(rep) >= {"discrimination", "benchmark", "scenarios"}
    assert set(rep["discrimination"]) == {"day2_known_auc", "day2_known_count", "day2_known_bad_rate"}
    assert set(rep["benchmark"]) == {
        "v3a_approval_rate", "v3a_bad_rate", "cle_approval_rate", "cle_bad_rate",
    }
    assert set(rep["scenarios"]) == {
        "1_no_ko_match_bad_rate", "2_income500_match_bad_rate", "3_v3a_ko_match_bad_rate",
        "4_no_ko_match_approval_rate", "5_income500_match_approval_rate",
        "6_v3a_ko_match_approval_rate", "7_income500_broad_match_bad_rate",
    }
    sc = rep["scenarios"]["2_income500_match_bad_rate"]
    assert {"ltv_per_link_d90", "ltv_per_link_d120", "ko_gate", "objective", "tracks"} <= set(sc)
    assert set(sc["tracks"]) == {"uw", "cle"}
    uw = sc["tracks"]["uw"]
    assert {
        "candidate_score_cutoff", "candidate_approved_count", "candidate_approval_rate",
        "v3a_approval_rate", "approval_rate_delta", "candidate_bad_rate", "v3a_bad_rate",
        "bad_rate_delta", "swap_in_bad_rate", "swap_out_bad_rate",
        "swap_in_count", "swap_out_count", "day1_approval_rate",
    } == set(uw)
    assert uw["approval_rate_delta"] == (uw["candidate_approval_rate"] - uw["v3a_approval_rate"])


def test_decision_metrics_and_spec():
    from projects.neobank_ncm.eval import decision_eval_spec
    from projects.neobank_ncm.eval.metrics import DecisionReport, Day2KnownAuc

    daily = _scored_daily_with_ltv()
    y_pred = daily["v3_score"]

    auc_rec = Day2KnownAuc().evaluate(daily, y_pred, "went_dpd45")
    assert auc_rec["name"] == "day2_known_auc"
    assert isinstance(auc_rec["value"], float) and 0.0 <= auc_rec["value"] <= 1.0

    rep_rec = DecisionReport().evaluate(daily, y_pred, "went_dpd45")
    assert rep_rec["name"] == "decision_report"
    assert "scenarios" in rep_rec["value"]            # structured (non-scalar) survives

    spec = decision_eval_spec()
    assert spec.primary_name == "day2_known_auc"      # scalar primary
    out = spec.evaluate(daily, y_pred, "went_dpd45")
    assert out["primary"] == "day2_known_auc"
    names = {m["name"] for m in out["metrics"]}
    assert names == {"day2_known_auc", "decision_report"}
