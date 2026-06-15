import pandas as pd

from projects.fraud_anomaly_detection.neo4j_codex.control.plug_report import summarize_plugs


def test_summarize_plugs_splits_discovery_coverage_from_outside_discovery(tiny_store):
    burned = pd.DataFrame({"entity_type": ["device"], "entity_value": ["devX"]})

    report = summarize_plugs(
        tiny_store,
        burned,
        discovery_users=["u1", "u2", "u3"],
    )

    assert report["covered_discovery"]["n_users"] == 0
    assert report["outside_discovery"]["n_users"] == 2
    assert report["outside_discovery"]["outcomes"]["n_dpd45_advances"] == 0
    assert report["uncovered_discovery"]["n_users"] == 3


def test_summarize_plugs_can_measure_holdout_delta_only(tiny_store):
    burned = pd.DataFrame({"entity_type": ["bank"], "entity_value": ["acctA"]})

    report = summarize_plugs(
        tiny_store,
        burned,
        discovery_users=["u3", "u4"],
        eligible_users=["u3", "u4"],
        start_ts=pd.Timestamp("2026-01-21"),
    )

    assert report["covered_discovery"]["n_users"] == 1
    assert report["covered_discovery"]["outcomes"]["n_dpd45_advances"] == 1
    assert report["outside_discovery"]["n_users"] == 0
    assert report["uncovered_discovery"]["n_users"] == 1
