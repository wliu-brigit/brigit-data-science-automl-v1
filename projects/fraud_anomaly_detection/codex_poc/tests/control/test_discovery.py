from pathlib import Path

import pytest

from projects.fraud_anomaly_detection.codex_poc.control.discovery import DiscoveryMethod
from projects.fraud_anomaly_detection.codex_poc.control.discovery.catalog import default_methods
from projects.fraud_anomaly_detection.codex_poc.control.discovery.graph_method import (
    ResidualRingMethod,
)
from projects.fraud_anomaly_detection.codex_poc.control.discovery.metadata import (
    MethodMetadata,
)
from projects.fraud_anomaly_detection.codex_poc.control.discovery.scenario_method import (
    ScenarioMethod,
)

SAMPLE = Path("projects/fraud_anomaly_detection/data/graph/fraud_graph.duckdb")


def test_default_method_catalog_is_the_extension_point():
    methods = default_methods()

    assert [method.name for method in methods] == [
        "scenario:ring_account_reuse",
        "graph:residual_ring_members",
    ]
    assert all(isinstance(method, DiscoveryMethod) for method in methods)
    metadata = [method.metadata for method in methods]
    assert [meta.name for meta in metadata] == [
        "scenario:ring_account_reuse",
        "graph:residual_ring_members",
    ]
    assert metadata[0].method_type == "scenario"
    assert metadata[0].time_semantics == "production_safe"
    assert metadata[0].promotion_tier == "plug_candidate"
    assert metadata[0].enforcement_projection == "scenario_rule"
    assert metadata[1].method_type == "graph"
    assert metadata[1].time_semantics == "snapshot_review"
    assert metadata[1].promotion_tier == "review_queue"
    assert metadata[1].enforcement_projection == "entity_key"


def test_method_catalog_can_filter_enabled_methods():
    methods = default_methods(enabled_only=True)

    assert all(method.metadata.enabled for method in methods)
    assert [method.metadata.name for method in methods] == [
        "scenario:ring_account_reuse",
        "graph:residual_ring_members",
    ]


def test_method_metadata_params_are_immutable():
    metadata = MethodMetadata(
        name="graph:test",
        version="v1",
        method_type="graph",
        time_semantics="snapshot_review",
        promotion_tier="review_queue",
        enforcement_projection="entity_key",
        params={"source": "test"},
    )

    with pytest.raises(TypeError):
        metadata.params["source"] = "changed"


def test_scenario_method_is_a_discovery_method():
    method = ScenarioMethod(scenario_name="ring_account_reuse")

    assert isinstance(method, DiscoveryMethod)
    assert method.name == "scenario:ring_account_reuse"
    assert method.metadata.params == {"scenario_name": "ring_account_reuse"}


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample store not built")
def test_scenario_method_emits_contract_findings_on_sample():
    finding_set = ScenarioMethod("ring_account_reuse").run(SAMPLE)

    assert finding_set.method == "scenario:ring_account_reuse"
    assert finding_set.method_version
    assert len(finding_set.findings) > 0
    for finding in finding_set.findings:
        assert finding.evidence["scenario"] == "ring_account_reuse"


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample store not built")
def test_graph_method_emits_contract_findings_on_sample():
    finding_set = ResidualRingMethod().run(SAMPLE)

    assert finding_set.method == "graph:residual_ring_members"
    assert len(finding_set.findings) > 0
    finding = finding_set.findings[0]
    assert isinstance(finding.user_id, str)
    assert "ring_users" in finding.evidence
    assert "entity_types" in finding.evidence
