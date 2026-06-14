"""Discovery method catalog for the walking skeleton.

The catalog is the reviewed extension point for methods that are live in the
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


def all_methods() -> list[DiscoveryMethod]:
    """Return every reviewed discovery method known to the skeleton."""
    return [ScenarioMethod("ring_account_reuse"), ResidualRingMethod()]


def default_methods(*, enabled_only: bool = True) -> list[DiscoveryMethod]:
    """Return discovery methods enabled for the default skeleton profile."""
    methods = all_methods()
    if not enabled_only:
        return methods
    return [method for method in methods if method.metadata.enabled]
