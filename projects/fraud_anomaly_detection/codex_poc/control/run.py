"""Walking-skeleton orchestrator: discovery -> findings -> plugs -> monitoring."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from projects.fraud_anomaly_detection.codex_poc.control import holdout, monitoring, plug
from projects.fraud_anomaly_detection.codex_poc.control.config import ControlConfig
from projects.fraud_anomaly_detection.codex_poc.control.discovery import DiscoveryMethod
from projects.fraud_anomaly_detection.codex_poc.control.discovery.catalog import default_methods
from projects.fraud_anomaly_detection.codex_poc.control.finding_store import FindingStore


def run_skeleton(
    store: Path | str,
    findings_db: Path | str,
    config: ControlConfig = ControlConfig(),
    refresh_key: str = "skeleton",
    methods: Sequence[DiscoveryMethod] | None = None,
) -> dict:
    active_methods = list(methods) if methods is not None else default_methods()
    finding_sets = [method.run(store) for method in active_methods]
    finding_store = FindingStore(findings_db)
    data_version = "sample"
    finding_store.write_snapshot(refresh_key, data_version=data_version, finding_sets=finding_sets)
    findings = finding_store.read_latest()

    split = holdout.two_state_split(store, config)
    discovered_a = findings.loc[findings["user_id"].isin(split.state_a_users), "user_id"]
    stats = plug.candidate_stats(store, discovered_a, eligible_users=split.state_a_users)
    burned = plug.qualify(stats, config)
    holdout_report = monitoring.holdout_effect(store, burned, split.holdout_users)
    burned_keys = burned[
        ["entity_type", "entity_value", "dpd45_precision", "coverage", "support"]
    ].to_dict("records")
    return {
        "discovery": {
            "methods": [
                {
                    "method": finding_set.method,
                    "method_version": finding_set.method_version,
                    "findings": len(finding_set.findings),
                }
                for finding_set in finding_sets
            ],
            "n_users": int(findings["user_id"].nunique()),
        },
        "finding_store": {
            "refresh_key": refresh_key,
            "data_version": data_version,
            "n_rows": int(len(findings)),
            "n_users": int(findings["user_id"].nunique()),
        },
        "plug": {
            "candidate_count": int(len(stats)),
            "burned_key_count": int(len(burned)),
            "burned_keys": burned_keys,
        },
        "holdout": holdout_report,
        # Backward-compatible top-level summary for the original smoke test.
        "n_findings": int(findings["user_id"].nunique()),
        "burned_keys": burned_keys,
    }
