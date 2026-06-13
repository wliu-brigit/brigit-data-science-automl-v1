import pandas as pd

from projects.fraud_anomaly_detection.codex_poc.control.outcomes import summarize_users


def test_summarize_users_counts_advance_grain_outcomes(tiny_store):
    summary = summarize_users(tiny_store, ["u1", "u2", "u3"])

    assert summary == {
        "n_users": 3,
        "n_users_with_advances": 3,
        "n_advances": 3,
        "n_mature_advances": 3,
        "n_dpd45_advances": 3,
        "dpd45_advance_rate": 1.0,
        "n_dpd45_users": 3,
        "dpd45_user_rate": 1.0,
    }


def test_summarize_users_can_measure_only_holdout_advances(tiny_store):
    summary = summarize_users(
        tiny_store,
        ["u1", "u3"],
        start_ts=pd.Timestamp("2026-01-21"),
    )

    assert summary["n_users"] == 2
    assert summary["n_users_with_advances"] == 1
    assert summary["n_advances"] == 1
    assert summary["n_dpd45_advances"] == 1
    assert summary["dpd45_advance_rate"] == 1.0
