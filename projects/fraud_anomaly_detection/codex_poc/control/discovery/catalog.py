"""Discovery method catalog for the walking skeleton.

This is the reviewed, in-repo extension point for methods that are live in the
control loop. Scenario definitions still live in ``scenarios/register.yaml``;
graph methods live as adapters under ``control.discovery``.
"""
from __future__ import annotations

from projects.fraud_anomaly_detection.codex_poc.control.discovery import DiscoveryMethod
from projects.fraud_anomaly_detection.codex_poc.control.discovery.graph_method import (
    ResidualRingMethod,
)
from projects.fraud_anomaly_detection.codex_poc.control.discovery.scenario_method import (
    ScenarioMethod,
)


def default_methods() -> list[DiscoveryMethod]:
    """Return the discovery methods currently wired into the skeleton run."""
    return [ScenarioMethod("ring_account_reuse"), ResidualRingMethod()]
