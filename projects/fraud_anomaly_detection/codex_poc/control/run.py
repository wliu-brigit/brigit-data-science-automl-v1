"""Walking-skeleton orchestrator: discovery -> findings -> plugs -> monitoring."""
from __future__ import annotations

from pathlib import Path

from projects.fraud_anomaly_detection.codex_poc.control import holdout, monitoring, plug
from projects.fraud_anomaly_detection.codex_poc.control.config import ControlConfig
from projects.fraud_anomaly_detection.codex_poc.control.discovery.graph_method import (
    ResidualRingMethod,
)
from projects.fraud_anomaly_detection.codex_poc.control.discovery.scenario_method import (
    ScenarioMethod,
)
from projects.fraud_anomaly_detection.codex_poc.control.finding_store import FindingStore

METHODS = [ScenarioMethod("ring_account_reuse"), ResidualRingMethod()]


def run_skeleton(
    store: Path | str,
    findings_db: Path | str,
    config: ControlConfig = ControlConfig(),
    refresh_key: str = "skeleton",
) -> dict:
    finding_sets = [method.run(store) for method in METHODS]
    finding_store = FindingStore(findings_db)
    finding_store.write_snapshot(refresh_key, data_version="sample", finding_sets=finding_sets)
    findings = finding_store.read_latest()

    split = holdout.two_state_split(store, config)
    discovered_a = findings.loc[findings["user_id"].isin(split.state_a_users), "user_id"]
    stats = plug.candidate_stats(store, discovered_a, eligible_users=split.state_a_users)
    burned = plug.qualify(stats, config)
    holdout_report = monitoring.holdout_effect(store, burned, split.holdout_users)
    return {
        "n_findings": int(findings["user_id"].nunique()),
        "burned_keys": burned[
            ["entity_type", "entity_value", "dpd45_precision", "coverage", "support"]
        ].to_dict("records"),
        "holdout": holdout_report,
    }
