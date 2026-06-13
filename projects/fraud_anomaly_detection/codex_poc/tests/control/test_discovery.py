from pathlib import Path

import pytest

from projects.fraud_anomaly_detection.codex_poc.control.discovery import DiscoveryMethod
from projects.fraud_anomaly_detection.codex_poc.control.discovery.graph_method import (
    ResidualRingMethod,
)
from projects.fraud_anomaly_detection.codex_poc.control.discovery.scenario_method import (
    ScenarioMethod,
)

SAMPLE = Path("projects/fraud_anomaly_detection/data/graph/fraud_graph.duckdb")


def test_scenario_method_is_a_discovery_method():
    method = ScenarioMethod(scenario_name="ring_account_reuse")

    assert isinstance(method, DiscoveryMethod)
    assert method.name == "scenario:ring_account_reuse"


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
