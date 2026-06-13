from projects.fraud_anomaly_detection.codex_poc.control.contract import Finding, FindingSet
from projects.fraud_anomaly_detection.codex_poc.control.config import ControlConfig


def test_finding_set_to_frame_has_contract_columns():
    fs = FindingSet(
        method="scenario:ring_device_burst",
        method_version="2026-06-08.2",
        findings=[Finding(user_id="u1", score=1.0, evidence={"scenario": "ring_device_burst"})],
    )
    df = fs.to_frame()
    assert list(df.columns) == ["user_id", "method", "method_version", "score", "evidence"]
    assert df.iloc[0]["user_id"] == "u1"
    assert df.iloc[0]["method"] == "scenario:ring_device_burst"


def test_config_defaults_are_tunable_in_one_place():
    cfg = ControlConfig()
    assert cfg.block_tier_precision == 0.8
    assert cfg.min_support >= 1
    assert cfg.min_coverage >= 1
    assert cfg.min_corroborating_types >= 1
    assert cfg.holdout_days == 30
    cfg2 = ControlConfig(block_tier_precision=0.7)
    assert cfg2.block_tier_precision == 0.7
