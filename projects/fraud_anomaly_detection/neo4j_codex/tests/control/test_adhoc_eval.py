"""Quick tests for the ad-hoc evaluator, shared metrics, and the discovery cache."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from projects.fraud_anomaly_detection.neo4j_codex.control import control_loop_report as clr
from projects.fraud_anomaly_detection.neo4j_codex.control.discovery import adhoc_eval, metrics


def _truth(tiny_store):
    return metrics.user_truth(metrics.load_advances(tiny_store))


def test_outcome_denominator_is_whole_set(tiny_store):
    # u1,u2,u3 are DPD45; u4,u5 are mature-but-clean. Rate is over ALL 5 users.
    o = metrics.outcome(["u1", "u2", "u3", "u4", "u5"], _truth(tiny_store))
    assert o["users"] == 5
    assert o["dpd45_users"] == 3
    assert o["dpd45_user_rate"] == pytest.approx(0.6)


def test_outcome_empty_is_zeroed(tiny_store):
    assert metrics.outcome([], _truth(tiny_store)) == metrics.empty_outcome()


def test_outcome_counts_unknown_users_in_denominator_only(tiny_store):
    # Unknown users have no scorable advance: counted in `users`, not `users_with_advances`.
    o = metrics.outcome(["u1", "nope"], _truth(tiny_store))
    assert o["users"] == 2
    assert o["users_with_advances"] == 1
    assert o["dpd45_users"] == 1
    assert o["dpd45_user_rate"] == pytest.approx(0.5)
    assert o["dpd45_user_rate_with_advances"] == pytest.approx(1.0)


def test_user_truth_window_filters_advances(tiny_store):
    # u1 transacted 2026-01-01 (before cutoff), u3 on 2026-02-15 (after) — only u3 counts.
    truth = metrics.user_truth(
        metrics.load_advances(tiny_store), start=pd.Timestamp("2026-01-21")
    )
    o = metrics.outcome(["u1", "u3"], truth)
    assert o["users"] == 2
    assert o["users_with_advances"] == 1
    assert o["advances"] == 1
    assert o["dpd45_advances"] == 1
    assert o["dpd45_advance_rate"] == pytest.approx(1.0)


def test_evaluate_candidate_no_cache(tiny_store):
    result = adhoc_eval.evaluate_candidate(
        users={"u1", "u2", "u3"}, store=tiny_store, cache=None, name="c"
    )
    assert result["n_candidate_users"] == 3
    assert result["n_users_off_store"] == 0
    assert result["candidate"]["dpd45_user_rate"] == pytest.approx(1.0)
    assert result["net_new"] is None


def test_evaluate_candidate_excludes_off_store_users(tiny_store):
    result = adhoc_eval.evaluate_candidate(
        users={"u1", "u2", "zzz"}, store=tiny_store, cache=None, name="c"
    )
    assert result["n_candidate_users"] == 3
    assert result["n_users_off_store"] == 1
    assert result["candidate"]["users"] == 2  # only u1,u2 scored


def test_evaluate_candidate_net_new_and_overlap(tiny_store):
    cache = {
        "store": str(tiny_store),
        "scenario_version": "x",
        "final_discovery": ["u1", "u2"],
        "methods": {"m1": ["u1"]},
    }
    result = adhoc_eval.evaluate_candidate(
        users={"u1", "u2", "u3"}, store=tiny_store, cache=cache, name="c"
    )
    assert result["net_new"]["n_net_new_users"] == 1  # u3 only
    assert result["net_new"]["outcomes"]["dpd45_user_rate"] == pytest.approx(1.0)
    assert result["cache"]["stale"] is False
    overlap = {row["method"]: row for row in result["per_method"]}
    assert overlap["m1"]["overlap_users"] == 1
    assert overlap["m1"]["net_new_beyond_method"] == 2
    assert adhoc_eval.render_markdown(result)  # renders without error


def test_discovery_cache_roundtrip(tmp_path):
    config = clr.ControlLoopReportConfig(out_dir=tmp_path, refresh_key="rk")
    scenario = SimpleNamespace(name="scenario:s1", users={"u1", "u2"})
    graph = SimpleNamespace(name="graph:g1", users={"u2", "u3"})
    path = clr._write_discovery_cache(
        config,
        scenario_candidates=[scenario],
        graph_rows=[graph],
        scenario_union={"u1", "u2"},
        final_discovery={"u1", "u2", "u3"},
    )
    assert path == clr.discovery_cache_path(config)
    cache = adhoc_eval.load_cache(path)
    assert sorted(cache["final_discovery"]) == ["u1", "u2", "u3"]
    assert cache["methods"]["scenario:s1"] == ["u1", "u2"]
    assert cache["methods"]["graph:g1"] == ["u2", "u3"]


def test_load_cache_missing_returns_none(tmp_path):
    assert adhoc_eval.load_cache(tmp_path / "nope.json") is None
