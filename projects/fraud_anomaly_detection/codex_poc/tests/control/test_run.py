from pathlib import Path
from datetime import datetime

import pytest
import duckdb

from projects.fraud_anomaly_detection.codex_poc.control.contract import Finding, FindingSet
from projects.fraud_anomaly_detection.codex_poc.control.config import ControlConfig
from projects.fraud_anomaly_detection.codex_poc.control.plug_fact_store import (
    PlugFactStore,
)
from projects.fraud_anomaly_detection.codex_poc.control.report_store import ReportStore
from projects.fraud_anomaly_detection.codex_poc.control.discovery.metadata import (
    MethodMetadata,
)
from projects.fraud_anomaly_detection.codex_poc.control.run import run_skeleton

SAMPLE = Path("projects/fraud_anomaly_detection/data/graph/fraud_graph.duckdb")


class StaticMethod:
    name = "test:static"
    metadata = MethodMetadata(
        name="test:static",
        version="v1",
        method_type="model",
        time_semantics="leakfree_asof",
        promotion_tier="plug_candidate",
        enforcement_projection="entity_key",
    )

    def run(self, store):
        return FindingSet(
            method=self.name,
            method_version=self.metadata.version,
            findings=[Finding("u1"), Finding("u2"), Finding("u3")],
        )


class LegacyMethodWithoutMetadata:
    name = "test:legacy"

    def run(self, store):
        return FindingSet(
            method=self.name,
            method_version="legacy",
            findings=[Finding("u1")],
        )


class FakeMetadataMethod:
    name = "test:fake"
    metadata = object()

    def run(self, store):
        return FindingSet(
            method=self.name,
            method_version="fake",
            findings=[Finding("u1")],
        )


class MismatchedFindingSetMethod:
    name = "test:mismatch"
    metadata = MethodMetadata(
        name="test:mismatch",
        version="v1",
        method_type="model",
        time_semantics="leakfree_asof",
        promotion_tier="plug_candidate",
        enforcement_projection="entity_key",
    )

    def run(self, store):
        return FindingSet(
            method="test:other",
            method_version="v1",
            findings=[Finding("u1")],
        )


class MismatchedVersionMethod:
    name = "test:version"
    metadata = MethodMetadata(
        name="test:version",
        version="v2",
        method_type="model",
        time_semantics="leakfree_asof",
        promotion_tier="plug_candidate",
        enforcement_projection="entity_key",
    )

    def run(self, store):
        return FindingSet(
            method=self.name,
            method_version="v1",
            findings=[Finding("u1")],
        )


class ShadowedFindingSetMethod:
    name = "test:shadow"
    metadata = MethodMetadata(
        name="test:shadow",
        version="v1",
        method_type="model",
        time_semantics="leakfree_asof",
        promotion_tier="plug_candidate",
        enforcement_projection="entity_key",
    )

    def run(self, store):
        return FindingSet(
            method=StaticMethod.name,
            method_version=StaticMethod.metadata.version,
            findings=[Finding("u1")],
        )


class ReviewQueueMethod:
    name = "test:review"
    metadata = MethodMetadata(
        name="test:review",
        version="v1",
        method_type="graph",
        time_semantics="snapshot_review",
        promotion_tier="review_queue",
        enforcement_projection="entity_key",
    )

    def run(self, store):
        return FindingSet(
            method=self.name,
            method_version=self.metadata.version,
            findings=[Finding("u1"), Finding("u2")],
        )


class FutureSensitiveMethod:
    name = "test:future_sensitive"
    metadata = MethodMetadata(
        name="test:future_sensitive",
        version="v1",
        method_type="model",
        time_semantics="leakfree_asof",
        promotion_tier="plug_candidate",
        enforcement_projection="entity_key",
    )

    def run(self, store):
        with duckdb.connect(str(store), read_only=True) as con:
            newest = con.execute("SELECT max(feature_as_of_ts) FROM advances").fetchone()[0]
        findings = [Finding("u1")] if newest >= datetime(2026, 2, 1) else []
        return FindingSet(
            method=self.name,
            method_version=self.metadata.version,
            findings=findings,
        )


class DuplicateNamePlugMethod:
    name = "test:duplicate"
    metadata = MethodMetadata(
        name="test:duplicate",
        version="plug",
        method_type="model",
        time_semantics="leakfree_asof",
        promotion_tier="plug_candidate",
        enforcement_projection="entity_key",
    )

    def run(self, store):
        return FindingSet(
            method=self.name,
            method_version=self.metadata.version,
            findings=[Finding("u1")],
        )


class DuplicateNameReviewMethod:
    name = "test:duplicate"
    metadata = MethodMetadata(
        name="test:duplicate",
        version="review",
        method_type="graph",
        time_semantics="snapshot_review",
        promotion_tier="review_queue",
        enforcement_projection="entity_key",
    )

    def run(self, store):
        return FindingSet(
            method=self.name,
            method_version=self.metadata.version,
            findings=[Finding("u1")],
        )


def test_run_skeleton_rejects_methods_without_metadata(tiny_store, tmp_path):
    with pytest.raises(TypeError, match="metadata"):
        run_skeleton(
            tiny_store,
            findings_db=tmp_path / "findings.duckdb",
            methods=[LegacyMethodWithoutMetadata()],
        )


def test_run_skeleton_rejects_non_method_metadata(tiny_store, tmp_path):
    with pytest.raises(TypeError, match="MethodMetadata"):
        run_skeleton(
            tiny_store,
            findings_db=tmp_path / "findings.duckdb",
            methods=[FakeMetadataMethod()],
        )


def test_run_skeleton_rejects_finding_set_that_does_not_match_metadata(tiny_store, tmp_path):
    with pytest.raises(ValueError, match="does not match method order"):
        run_skeleton(
            tiny_store,
            findings_db=tmp_path / "findings.duckdb",
            methods=[MismatchedFindingSetMethod()],
        )


def test_run_skeleton_rejects_finding_set_version_that_does_not_match_metadata(
    tiny_store,
    tmp_path,
):
    with pytest.raises(ValueError, match="does not match metadata version"):
        run_skeleton(
            tiny_store,
            findings_db=tmp_path / "findings.duckdb",
            methods=[MismatchedVersionMethod()],
        )


def test_run_skeleton_rejects_finding_set_that_shadows_registered_method(
    tiny_store,
    tmp_path,
):
    with pytest.raises(ValueError, match="does not match method order"):
        run_skeleton(
            tiny_store,
            findings_db=tmp_path / "findings.duckdb",
            methods=[ShadowedFindingSetMethod(), StaticMethod()],
        )


def test_run_skeleton_rejects_duplicate_method_names(tiny_store, tmp_path):
    with pytest.raises(ValueError, match="Duplicate discovery method name"):
        run_skeleton(
            tiny_store,
            findings_db=tmp_path / "findings.duckdb",
            methods=[DuplicateNamePlugMethod(), DuplicateNameReviewMethod()],
        )


def test_run_skeleton_does_not_derive_plugs_from_review_queue_methods(tiny_store, tmp_path):
    report = run_skeleton(
        tiny_store,
        findings_db=tmp_path / "findings.duckdb",
        config=ControlConfig(min_support=2, min_coverage=1, block_tier_precision=0.5),
        methods=[ReviewQueueMethod()],
    )

    assert report["discovery"]["union"]["n_users"] == 2
    assert report["plug"]["candidate_count"] == 0
    assert report["plug"]["burned_key_count"] == 0
    assert report["state_a_backtest"]["plug"]["covered_discovery"]["n_users"] == 0


def test_run_skeleton_runs_state_a_discovery_on_asof_store(tiny_store, tmp_path):
    report = run_skeleton(
        tiny_store,
        findings_db=tmp_path / "findings.duckdb",
        config=ControlConfig(min_support=1, min_coverage=1, block_tier_precision=0.5),
        methods=[FutureSensitiveMethod()],
    )

    assert report["discovery"]["union"]["n_users"] == 1
    assert report["state_a_backtest"]["discovery"]["union"]["n_users"] == 0
    assert report["plug"]["candidate_count"] == 0


def test_run_skeleton_accepts_methods_and_returns_holistic_stage_report(tiny_store, tmp_path):
    findings_db = tmp_path / "findings.duckdb"
    report = run_skeleton(
        tiny_store,
        findings_db=findings_db,
        reports_db=tmp_path / "reports.duckdb",
        config=ControlConfig(min_support=2, min_coverage=1, block_tier_precision=0.5),
        methods=[StaticMethod()],
    )

    assert report["discovery"]["methods"][0]["method"] == "test:static"
    assert report["discovery"]["methods"][0]["findings"] == 3
    assert report["discovery"]["union"]["n_users"] == 3
    assert report["finding_store"] == {
        "refresh_key": "skeleton",
        "data_version": "sample",
        "snapshot_written": True,
        "snapshot_refresh_key": "skeleton",
        "snapshot_id": report["finding_store"]["snapshot_id"],
        "n_rows": 3,
        "n_users": 3,
    }
    assert report["plug"]["candidate_count"] >= 1
    assert report["plug"]["candidate_fact_snapshot_id"]
    assert report["plug"]["burned_key_count"] >= 1
    assert report["state_a_backtest"]["discovery"]["union"]["n_users"] == 2
    assert report["state_a_backtest"]["plug"]["covered_discovery"]["n_users"] == 2
    assert report["state_a_backtest"]["plug"]["outside_discovery"]["n_users"] == 0
    assert report["holdout_backtest"]["discovery"]["union"]["n_users"] == 1
    assert report["holdout_backtest"]["plug"]["covered_discovery"]["n_users"] == 1
    assert report["holdout_backtest"]["plug"]["outside_discovery"]["n_users"] == 0

    latest = ReportStore(tmp_path / "reports.duckdb").read_latest()
    assert latest["refresh_key"] == "skeleton"
    assert latest["report"]["holdout_backtest"] == report["holdout_backtest"]
    plug_facts = PlugFactStore(findings_db).read_latest()
    assert len(plug_facts) == report["plug"]["candidate_count"]


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample store not built")
def test_run_skeleton_end_to_end(tmp_path):
    report = run_skeleton(
        SAMPLE,
        findings_db=tmp_path / "findings.duckdb",
        config=ControlConfig(min_support=2, min_coverage=1, block_tier_precision=0.5),
    )

    assert report["discovery"]["union"]["n_users"] > 0
    assert "burned_keys" in report["plug"]
    assert "holdout_backtest" in report
    assert "n_findings" not in report
    assert "burned_keys" not in report
    assert "holdout" not in report
    assert set(report["holdout_backtest"]["plug"]).issuperset(
        {"covered_discovery", "uncovered_discovery", "outside_discovery"}
    )
