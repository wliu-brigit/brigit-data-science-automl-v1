from pathlib import Path

import pytest

from projects.fraud_anomaly_detection.codex_poc.control.config import ControlConfig
from projects.fraud_anomaly_detection.codex_poc.control.run import run_skeleton

SAMPLE = Path("projects/fraud_anomaly_detection/data/graph/fraud_graph.duckdb")


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample store not built")
def test_run_skeleton_end_to_end(tmp_path):
    report = run_skeleton(
        SAMPLE,
        findings_db=tmp_path / "findings.duckdb",
        config=ControlConfig(min_support=2, min_coverage=1, block_tier_precision=0.5),
    )

    assert report["n_findings"] > 0
    assert "burned_keys" in report
    assert "holdout" in report
    assert set(report["holdout"]).issuperset({"prevented_bad", "leaked_bad"})
