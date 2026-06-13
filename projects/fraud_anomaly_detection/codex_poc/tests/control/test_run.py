from pathlib import Path

import pytest

from projects.fraud_anomaly_detection.codex_poc.control.contract import Finding, FindingSet
from projects.fraud_anomaly_detection.codex_poc.control.config import ControlConfig
from projects.fraud_anomaly_detection.codex_poc.control.run import run_skeleton

SAMPLE = Path("projects/fraud_anomaly_detection/data/graph/fraud_graph.duckdb")


class StaticMethod:
    name = "test:static"

    def run(self, store):
        return FindingSet(
            method=self.name,
            method_version="v1",
            findings=[Finding("u1"), Finding("u2"), Finding("u3")],
        )


def test_run_skeleton_accepts_methods_and_returns_holistic_stage_report(tiny_store, tmp_path):
    report = run_skeleton(
        tiny_store,
        findings_db=tmp_path / "findings.duckdb",
        config=ControlConfig(min_support=2, min_coverage=1, block_tier_precision=0.5),
        methods=[StaticMethod()],
    )

    assert report["discovery"]["methods"] == [
        {"method": "test:static", "method_version": "v1", "findings": 3}
    ]
    assert report["finding_store"] == {
        "refresh_key": "skeleton",
        "data_version": "sample",
        "n_rows": 3,
        "n_users": 3,
    }
    assert report["plug"]["candidate_count"] >= 1
    assert report["plug"]["burned_key_count"] >= 1
    assert report["holdout"]["prevented_bad"] == 1


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
