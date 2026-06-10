"""Offline tests for the downstream-analysis port (analysis/*).

Pins the ported legacy semantics — KO gating, the NaN-on-ineligible-days
min-score collapse, threshold matching, lookup imputation, the sample-size
math — against hand-built frames and the synthetic daily fixture, with a
stub model standing in for the scorer. No warehouse, no MLflow server.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projects.neobank_ncm.analysis import impact, policy, scoring
from projects.neobank_ncm.analysis.data import ri_scores_parity
from projects.neobank_ncm.tests.fixtures import (
    StubDailyModel,
    make_synthetic_daily_table,
    make_synthetic_ltv_table,
)


# ── derived features ─────────────────────────────────────────────────────────
def test_daily_derived_features_match_legacy_formulas():
    row = pd.DataFrame(
        {
            "dailyincomemean": [-10.0],  # abs() then eps clip
            "balancesd": [50.0],
            "maxnegativebalpast30days": [-30.0],
            "inflowsum14d": [-140.0],
            "outflowsum14d": [70.0],
            "balancemeanafterpayday0": [200.0],
            "balancemeanafterpayday1": [150.0],
            "highestpaydepositmean": [-25.0],
            "balancemean": [80.0],
            "daystopayday": [0.0],  # clipped to 1
            "davesummarycreditninetydayamount": [np.nan],  # fillna(0)
            "earninsummarycreditninetydayamount": [90.0],
            "othercompetitorsummarycreditninetydayamount": [np.nan],
            "snapshot_date": [pd.Timestamp("2026-02-10")],  # Feb -> tax season
        }
    )
    out = scoring.add_daily_derived_features(row.copy())
    assert out.loc[0, "balancesdtodailyincomemeanratio"] == pytest.approx(5.0)
    assert out.loc[0, "maxnegbalance30dtodailyincomemeanratio"] == pytest.approx(3.0)
    assert out.loc[0, "inflowsumtooutflowsumratio14d"] == pytest.approx(2.0)
    assert out.loc[0, "netflowtodailyincomemeanratio14d"] == pytest.approx(70.0 / 140.0)
    assert out.loc[0, "balancedepletionrate1d"] == pytest.approx(-2.0)
    assert out.loc[0, "incomebuffertodaystopaydayratio"] == pytest.approx(8.0)
    assert out.loc[0, "competitorborrowintensity"] == pytest.approx(90.0 / 900.0)
    # istaxseason anchors to SNAPSHOT date (the daily pipeline's difference)
    assert out.loc[0, "istaxseason"] == 1
    out_may = scoring.add_daily_derived_features(
        row.assign(snapshot_date=[pd.Timestamp("2026-05-10")]).copy()
    )
    assert out_may.loc[0, "istaxseason"] == 0


# ── policy: KO gating + collapse ─────────────────────────────────────────────
def _mini_daily() -> pd.DataFrame:
    """Three users, hand-built days; v3_score already attached."""

    def rows(user, entries, went_dpd45, synthetic):
        return [
            {
                "user_id": user,
                "day_number": day,
                "v3_score": score,
                "account_approval_state": state,
                "dailyincomemean": income / 30.0,
                "highestpaydepositmean": paycheck,
                "noactivityrate": activity,
                "v2_score": v2,
                "plaidfeaturessummary_incomewages_lookbackwindow30d_inflow_sum": 0.0,
                "went_dpd45": went_dpd45,
                "synthetic_score": synthetic,
                "is_known": went_dpd45 is not None and not pd.isna(went_dpd45),
            }
            for day, score, state, income, paycheck, activity, v2 in entries
        ]

    data = []
    # known user: best score (0.05) lands on a NOT_APPROVED day -> must not
    # count; KO fails on day 2 (income 600 < 700) so ko-min comes from day 3
    data += rows(
        "a",
        [
            (1, 0.05, "NOT_APPROVED", 800.0, 400.0, 0.5, 0.4),
            (2, 0.20, None, 600.0, 400.0, 0.5, 0.4),
            (3, 0.30, "APPROVED", 800.0, 400.0, 0.5, 0.4),
        ],
        went_dpd45=0.0,
        synthetic=0.10,
    )
    # unknown user: v3a passes on day 2 (v2 .40 <= .485, KOs pass, eligible)
    data += rows(
        "b",
        [
            (1, 0.50, None, 800.0, 400.0, 0.9, 0.40),  # KO fails (activity)
            (2, 0.40, None, 800.0, 400.0, 0.5, 0.40),
        ],
        went_dpd45=np.nan,
        synthetic=0.25,
    )
    # unknown user with no synthetic score -> dropped from user_scores
    data += rows(
        "c",
        [(1, 0.60, None, 800.0, 400.0, 0.5, 0.6)],
        went_dpd45=np.nan,
        synthetic=np.nan,
    )
    return pd.DataFrame(data)


def test_policy_columns_and_collapse_semantics():
    daily = policy.add_policy_columns(_mini_daily())
    users = policy.collapse_to_users(daily).set_index("user_id")

    # user c had no outcome and no synthetic score -> dropped
    assert list(users.index) == ["a", "b"]

    a = users.loc["a"]
    # NOT_APPROVED day's 0.05 never contributes; eligible min is 0.20
    assert a["min_v3_score"] == pytest.approx(0.20)
    # KO-eligible min skips the failing day 2 -> 0.30
    assert a["min_v3_score_ko"] == pytest.approx(0.30)
    # D1 snapshot keeps the RAW score even on an acct-ineligible day — the
    # legacy's executed behavior (its null-out assign was dead code)
    assert a["d1_v3_score"] == pytest.approx(0.05)
    assert a["effective_bad"] == pytest.approx(0.0)  # known: ground truth wins

    b = users.loc["b"]
    assert bool(b["v3a_approved"])  # day 2 passes all v3a gates
    assert b["effective_bad"] == pytest.approx(0.25)  # unknown: synthetic
    assert b["min_v3_score_ko"] == pytest.approx(0.40)  # day 1 KO-failed


# ── thresholds ───────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def _fixture_users_cached():
    daily = make_synthetic_daily_table()
    scoring.score_daily(daily, StubDailyModel())
    policy.add_policy_columns(daily)
    users = policy.collapse_to_users(daily)
    return daily, users


@pytest.fixture
def fixture_users(_fixture_users_cached):
    # per-test copies: score_daily/add_policy_columns mutate in place, so the
    # cached frames must never be shared mutably across tests
    daily, users = _fixture_users_cached
    return daily.copy(), users.copy()


def test_threshold_for_ar_hits_target(fixture_users):
    _, users = fixture_users
    n = len(users)
    target = 0.30
    t = policy.threshold_for_ar(users["min_v3_score"].values, target, n)
    achieved = float((users["min_v3_score"] <= t).sum()) / n
    assert achieved == pytest.approx(target, abs=2.0 / n)


def test_threshold_for_br_matches_target(fixture_users):
    _, users = fixture_users
    t = policy.threshold_for_br(users, 0.30, "min_v3_score")
    approved = users[users["min_v3_score"] <= t]
    achieved = approved["effective_bad"].mean()
    # the search picks the closest reachable cumulative bad rate
    assert achieved == pytest.approx(0.30, abs=0.05)


def test_extended_metrics_swap_accounting():
    users = pd.DataFrame(
        {
            "is_known": [True, True, False, True],
            "went_dpd45": [1.0, 0.0, np.nan, 0.0],
            "effective_bad": [1.0, 0.0, 0.5, 0.0],
        }
    )
    ref = pd.Series([True, True, False, False])
    new = pd.Series([True, False, True, False])
    m = policy.extended_metrics(ref, new, users)
    assert m["n_v3"] == 2
    assert m["v3_ar"] == pytest.approx(0.5)
    assert m["ref_ar"] == pytest.approx(0.5)
    assert m["v3_br"] == pytest.approx((1.0 + 0.5) / 2)
    assert m["ref_br"] == pytest.approx(0.5)  # known-only ground truth
    assert m["swap_in_vol"] == 1 and m["swap_out_vol"] == 1
    assert m["swap_in_br"] == pytest.approx(0.5)
    assert m["swap_out_br"] == pytest.approx(0.0)


def test_all_threshold_tables_shape(fixture_users):
    _, users = fixture_users
    bench = policy.benchmarks(users)
    thresholds = policy.compute_thresholds(users, bench)
    tables = policy.all_threshold_tables(users, thresholds)
    assert set(tables) == {
        f"{variant}/match_{objective}"
        for variant in ("no_ko", "income500", "income500_broad", "v3a_ko")
        for objective in ("ar", "br")
    }
    for table in tables.values():
        assert list(table.columns) == policy.TABLE_COLUMNS
        assert len(table) == 2
        assert table["n_v3"].ge(0).all()
        assert table["v3_thr"].notna().all()


def test_first_approval_and_curves(fixture_users):
    daily, users = fixture_users
    bench = policy.benchmarks(users)
    thresholds = policy.compute_thresholds(users, bench)
    scen = policy.scenario_map(thresholds)
    assert set(scen) == set(range(1, 8))
    _, thr_uw, thr_cle, ko_uw, ko_cle = scen[2]
    arrays = policy.first_approval_days(daily, users["user_id"], thr_uw, thr_cle, ko_uw, ko_cle)
    assert set(arrays) == {"uw", "cle", "v3", "v3a"}
    # combined track approves on the earlier of the two first-days
    assert np.all(
        np.isnan(arrays["v3"])
        | (arrays["v3"] == np.fmin(arrays["uw"], arrays["cle"]))
    )
    curves = policy.approval_curves(arrays)
    for col in curves.columns:
        assert (curves[col].diff().dropna() >= 0).all()  # cumulative => monotone


# ── impact: lookup + inference + sample size ─────────────────────────────────
@pytest.fixture
def fixture_us(_fixture_users_cached):
    _, users = _fixture_users_cached
    ltv = make_synthetic_ltv_table(users["user_id"])
    return impact.merge_ltv(users.copy(), ltv)


def test_lookup_imputation_and_actuals(fixture_us):
    us, lookup = impact.build_lookup(fixture_us)
    assert lookup.notna().all().all()  # sparse bins filled with global means

    approved = pd.Series(True, index=us.index)
    lam = pd.Series(50.0, index=us.index)
    inf = impact.infer_financials(us, lookup, approved, lam)

    mature = inf["is_activated"] & inf["ltv_90_elig"].astype(bool)
    # activated + mature keep actuals; everyone else gets the lookup
    assert np.allclose(
        inf.loc[mature, "inf_rev_90"], inf.loc[mature, "total_revenue_90"]
    )
    immature = ~mature
    looked_up = us.loc[immature, "score_bin"].map(lookup["rev_90"]).fillna(
        lookup["rev_90"].mean()
    )
    assert np.allclose(inf.loc[immature, "inf_rev_90"], looked_up)
    # imputed loss scales with the approval loan amount
    lr = us.loc[immature, "score_bin"].map(lookup["lr_90"]).fillna(lookup["lr_90"].mean())
    assert np.allclose(inf.loc[immature, "inf_loss_90"], lr * 50.0)


def test_monthly_aggregate_arithmetic(fixture_us):
    us, lookup = impact.build_lookup(fixture_us)
    inf = impact.infer_financials(
        us, lookup, us["v3a_approved"], pd.Series(50.0, index=us.index)
    )
    agg = impact.monthly_aggregate(inf, n=len(us), act_rate=0.4, monthly_links=1000.0)
    assert agg["mo_app"] == pytest.approx(agg["ar"] * 1000.0)
    assert agg["mo_act"] == pytest.approx(agg["mo_app"] * 0.4)
    assert agg["mo_ltv_90"] == pytest.approx(agg["mo_rev_90"] - agg["mo_loss_90"], rel=1e-9)


def test_sample_size_formulas():
    # AR test, hand-computed: p1=0.2, p2=0.3
    # h = 2(asin(sqrt(.3)) - asin(sqrt(.2))) = 0.231957..; n = (2.80158/h)^2
    us = pd.DataFrame(index=range(4))
    empty = pd.DataFrame(
        {"inf_ltv_90": [], "inf_ltv_120": []}, index=pd.Index([], dtype="int64")
    )
    result = impact.sample_size_analysis(
        us, empty, empty, p_control=0.2, p_treatment=0.3, monthly_vol=1000.0
    )
    h = 2 * (np.arcsin(np.sqrt(0.3)) - np.arcsin(np.sqrt(0.2)))
    expected_n = ((result["z_a"] + result["z_b"]) / h) ** 2
    assert result["n_per_test"]["(A) Approval rate"] == pytest.approx(expected_n)
    assert result["z_a"] == pytest.approx(1.959964, abs=1e-5)
    assert result["z_b"] == pytest.approx(0.841621, abs=1e-5)
    # LTV deltas are zero (empty arms) -> AR test binds
    assert result["binding"] == "(A) Approval rate"
    assert result["n_total"] == pytest.approx(2 * expected_n)
    months = result["months_table"]
    assert np.allclose(
        months["months_needed"], result["n_total"] / (1000.0 * months["holdout_rate"])
    )


def test_d1_proposal_mask_requires_ko_pass():
    users = pd.DataFrame(
        {
            "min_v3_score": [0.10, 0.10],
            "min_v3_score_ko500": [0.10, np.nan],  # second user never KO-passes
            "d1_v3_score": [0.10, 0.10],
            "d1_ko_pass_500": [False, np.nan],  # neither passes the KO on day 1
        }
    )
    approved, d1 = policy.proposal_masks(users, threshold=0.20, variant="k5")
    assert approved.tolist() == [True, False]  # KO-ineligible user blocked
    assert d1.tolist() == [False, False]  # low D1 score alone is not enough


def test_sample_size_ltv_branch_hand_computed():
    # ctrl zero-padded to [0,0,10,10]: mean 5, var(ddof=1)=100/3
    # trt  zero-padded to [0,2,10,12]: mean 6, var(ddof=1)=104/3 -> sigma2
    # delta = 1 -> n = 2*(104/3)*(z_a+z_b)^2
    us = pd.DataFrame(index=range(4))
    v3a_inf = pd.DataFrame({"inf_ltv_90": [10.0, 10.0], "inf_ltv_120": [10.0, 10.0]}, index=[2, 3])
    v3_inf = pd.DataFrame(
        {"inf_ltv_90": [2.0, 10.0, 12.0], "inf_ltv_120": [2.0, 10.0, 12.0]}, index=[1, 2, 3]
    )
    result = impact.sample_size_analysis(
        us, v3a_inf, v3_inf, p_control=0.2, p_treatment=0.2001, monthly_vol=100.0
    )
    t90 = result["ltv_tests"][90]
    assert t90["mu_ctrl"] == pytest.approx(5.0)
    assert t90["mu_trt"] == pytest.approx(6.0)
    assert t90["sigma2"] == pytest.approx(104.0 / 3.0)
    z_sum_sq = (result["z_a"] + result["z_b"]) ** 2
    assert result["n_per_test"]["(B) LTV/link D90"] == pytest.approx(2 * (104.0 / 3.0) * z_sum_sq)
    assert result["binding"].startswith("(B)") or result["binding"].startswith("(C)")


def test_revenue_decomposition_identity_when_fully_eligible():
    # crafted so every group is horizon-eligible -> the weighted effects sum
    # exactly to the total delta (legacy cell 24's identity)
    horizons = impact.HORIZONS
    act_us = pd.DataFrame(
        {
            **{f"total_revenue_{h}": [10.0, 20.0, 30.0] for h in horizons},
            **{f"ltv_{h}_elig": [True, True, True] for h in horizons},
        },
        index=[0, 1, 2],
    )
    v3_inf = pd.DataFrame(
        {
            "is_activated": [True, True, False, False],
            **{f"inf_rev_{h}": [20.0, 30.0, 8.0, 12.0] for h in horizons},
        },
        index=[1, 2, 4, 5],
    )
    decomp = impact.revenue_decomposition(act_us, v3_inf)
    assert decomp["counts"] == {"overlap": 2, "new": 0, "nonact": 2, "dropped": 1}
    row = decomp["decomposition"].loc[30]
    assert row["total_delta"] == pytest.approx(17.5 - 20.0)
    assert row["overlap_effect"] == pytest.approx(25.0 * (2 / 4 - 2 / 3))
    assert row["nonact_effect"] == pytest.approx(10.0 * 2 / 4)
    assert row["dropped_effect"] == pytest.approx(-10.0 / 3)
    assert row["check"] == pytest.approx(row["total_delta"])


def test_ri_scores_parity_perfect_agreement():
    scores = pd.DataFrame(
        {"user_id": [f"u{i}" for i in range(300)], "synthetic_score": np.linspace(0.01, 0.99, 300)}
    )
    final = scores.rename(columns={"synthetic_score": "final_score"})
    parity = ri_scores_parity(scores, final)
    assert parity["n_overlap"] == 300
    assert parity["pearson_corr"] == pytest.approx(1.0)
    assert parity["mean_abs_diff"] == pytest.approx(0.0)
    assert parity["pct_exact_percentile"] == pytest.approx(1.0)


# ── the eval script, end to end (file-backed MLflow, parquet inputs) ────────
def test_evaluate_script_end_to_end(tmp_path, monkeypatch, capsys):
    import importlib.util
    import shutil
    import sys
    from pathlib import Path

    project_dir = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "evaluate_new_links_daily",
        project_dir / "scripts" / "evaluate_new_links_daily.py",
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    daily = make_synthetic_daily_table(n_users=120, seed=23)
    daily_pq = tmp_path / "daily.parquet"
    daily.drop(columns=["is_known"]).to_parquet(daily_pq)  # loader re-derives it
    users = daily["user_id"].drop_duplicates()
    ri = pd.DataFrame({"user_id": users, "synthetic_score": np.linspace(0.01, 0.99, len(users))})
    ri_pq = tmp_path / "ri.parquet"
    ri.to_parquet(ri_pq)
    syn_pq = tmp_path / "syn.parquet"
    ri.rename(columns={"synthetic_score": "final_score"}).to_parquet(syn_pq)

    monkeypatch.setattr(script.scoring, "TrialModel", lambda run_id: StubDailyModel())
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path}/mlruns")
    namespace = "qa/analysis-script-test"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_new_links_daily.py",
            "--model-run-id",
            "stub",
            "--daily-parquet",
            str(daily_pq),
            "--ri-scores-parquet",
            str(ri_pq),
            "--syn-oot-parquet",
            str(syn_pq),
            "--namespace",
            namespace,
        ],
    )
    try:
        script.main()
    finally:
        # the script binds an active project session — clear it so this test
        # leaks nothing into later tests (e.g. "no active project" checks)
        from automl.project import clear_session

        clear_session()
        # use_project materializes local session state for the qa namespace
        shutil.rmtree(project_dir / "experiments" / namespace, ignore_errors=True)

    out = capsys.readouterr().out
    assert '"run_id"' in out
    assert '"d2_auc"' in out
    assert '"ri_parity_pearson_corr"' in out


# ── end-to-end smoke over the fixture ────────────────────────────────────────
def test_full_pipeline_smoke(fixture_users, fixture_us):
    daily, users = fixture_users

    # scoring QA pieces
    cal = scoring.calibration_table(daily)
    assert cal["n"].sum() > 0 and cal["delta"].abs().max() < 1.0
    auc = scoring.d2_known_auc(daily)
    assert auc["d2_auc"] > 0.55  # stub model recovers the planted signal

    bench = policy.benchmarks(users)
    thresholds = policy.compute_thresholds(users, bench)
    scen_label, thr_uw, thr_cle, ko_uw, ko_cle = policy.scenario_map(thresholds)[2]
    arrays = policy.first_approval_days(
        daily, users["user_id"], thr_uw, thr_cle, ko_uw, ko_cle
    )

    us, lookup = impact.build_lookup(fixture_us)
    ref = impact.historical_reference(us)
    assert 0.0 <= ref["act_rate"] <= 1.0

    uw_mask = us["user_id"].isin(set(users["user_id"][arrays["uw"] <= 30]))
    cle_mask = us["user_id"].isin(set(users["user_id"][arrays["cle"] <= 30]))
    cle_only = cle_mask & ~uw_mask
    v3_mask = uw_mask | cle_only
    lam_v3 = pd.Series(
        np.where(uw_mask, policy.LAM_UW, np.where(cle_only, policy.LAM_CLE, np.nan)),
        index=us.index,
    )
    v3_inf = impact.infer_financials(us, lookup, v3_mask, lam_v3)
    agg = impact.monthly_aggregate(v3_inf, len(us), ref["act_rate"], ref["monthly_vol"])
    assert agg["n_app"] == int(v3_mask.sum())

    decomp = impact.revenue_decomposition(ref["act_us"], v3_inf)
    counts = decomp["counts"]
    assert counts["overlap"] + counts["new"] + counts["nonact"] == decomp["n_v3"]
    assert decomp["decomposition"].notna().all().all()

    lam_all = us["loan_amount_max"].fillna(us["loan_amount_max"].median())
    v3a_inf = impact.infer_financials(us, lookup, us["v3a_approved"], lam_all)
    sizing = impact.sample_size_analysis(
        us,
        v3a_inf,
        v3_inf,
        p_control=bench["v3a_ar"],
        p_treatment=float((arrays["v3"] <= 30).mean()),
        monthly_vol=ref["monthly_vol"],
    )
    assert sizing["n_total"] >= 0
    assert (sizing["months_table"]["months_needed"] >= 0).all()
