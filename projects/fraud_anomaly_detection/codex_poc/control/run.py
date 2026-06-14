"""Walking-skeleton orchestrator: discovery -> findings -> plugs -> monitoring."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from projects.fraud_anomaly_detection.codex_poc.control import holdout, plug
from projects.fraud_anomaly_detection.codex_poc.control.config import ControlConfig
from projects.fraud_anomaly_detection.codex_poc.control.contract import FindingSet
from projects.fraud_anomaly_detection.codex_poc.control.discovery import DiscoveryMethod
from projects.fraud_anomaly_detection.codex_poc.control.discovery.catalog import default_methods
from projects.fraud_anomaly_detection.codex_poc.control.discovery.metadata import (
    MethodMetadata,
)
from projects.fraud_anomaly_detection.codex_poc.control.discovery_report import summarize_discovery
from projects.fraud_anomaly_detection.codex_poc.control.finding_store import FindingStore
from projects.fraud_anomaly_detection.codex_poc.control.plug_report import summarize_plugs
from projects.fraud_anomaly_detection.codex_poc.control.report_store import ReportStore


def run_skeleton(
    store: Path | str,
    findings_db: Path | str,
    reports_db: Path | str | None = None,
    config: ControlConfig = ControlConfig(),
    refresh_key: str = "skeleton",
    methods: Sequence[DiscoveryMethod] | None = None,
) -> dict:
    active_methods = list(methods) if methods is not None else default_methods()
    _validate_methods(active_methods)
    finding_sets = [method.run(store) for method in active_methods]
    _validate_finding_sets(finding_sets, active_methods)
    finding_store = FindingStore(findings_db)
    data_version = "sample"
    finding_store.write_snapshot(refresh_key, data_version=data_version, finding_sets=finding_sets)
    findings = finding_store.read_latest()

    split = holdout.two_state_split(store, config)
    full_discovery_report = summarize_discovery(store, finding_sets)
    state_a_discovery_report = summarize_discovery(
        store,
        finding_sets,
        eligible_users=split.state_a_users,
    )
    holdout_discovery_report = summarize_discovery(
        store,
        finding_sets,
        eligible_users=split.holdout_users,
        start_ts=split.cutoff,
    )

    plug_finding_sets = _finding_sets_for_promotion(finding_sets, active_methods)
    discovered_a = _finding_users(plug_finding_sets, eligible_users=split.state_a_users)
    stats = plug.candidate_stats(store, discovered_a, eligible_users=split.state_a_users)
    burned = plug.qualify(stats, config)
    state_a_plug_report = summarize_plugs(
        store,
        burned,
        discovery_users=discovered_a,
        eligible_users=split.state_a_users,
    )
    discovered_holdout = _finding_users(
        plug_finding_sets,
        eligible_users=split.holdout_users,
    )
    holdout_plug_report = summarize_plugs(
        store,
        burned,
        discovery_users=discovered_holdout,
        eligible_users=split.holdout_users,
        start_ts=split.cutoff,
    )
    burned_keys = burned[
        ["entity_type", "entity_value", "dpd45_precision", "coverage", "support"]
    ].to_dict("records")
    report = {
        "discovery": full_discovery_report,
        "finding_store": {
            "refresh_key": refresh_key,
            "data_version": data_version,
            "n_rows": int(len(findings)),
            "n_users": int(findings["user_id"].nunique()),
        },
        "state_a_backtest": {
            "split": {
                "cutoff": split.cutoff.isoformat(),
                "newest": split.newest.isoformat(),
                "n_users": len(split.state_a_users),
            },
            "discovery": state_a_discovery_report,
            "plug": state_a_plug_report,
        },
        "holdout_backtest": {
            "split": {
                "cutoff": split.cutoff.isoformat(),
                "newest": split.newest.isoformat(),
                "n_users": len(split.holdout_users),
            },
            "discovery": holdout_discovery_report,
            "plug": holdout_plug_report,
        },
        "run": {
            "refresh_key": refresh_key,
            "data_version": data_version,
        },
        "plug": {
            "candidate_count": int(len(stats)),
            "burned_key_count": int(len(burned)),
            "burned_keys": burned_keys,
            "validation": state_a_plug_report,
        },
    }
    if reports_db is not None:
        ReportStore(reports_db).write_report(refresh_key, data_version=data_version, report=report)
    return report


def _finding_users(
    finding_sets: Sequence[FindingSet],
    eligible_users: list[str] | None = None,
) -> pd.Series:
    eligible = None if eligible_users is None else {str(user_id) for user_id in eligible_users}
    users: set[str] = set()
    for finding_set in finding_sets:
        for finding in finding_set.findings:
            user_id = str(finding.user_id)
            if eligible is None or user_id in eligible:
                users.add(user_id)
    return pd.Series(sorted(users), dtype="string")


def _validate_methods(methods: Sequence[DiscoveryMethod]) -> None:
    seen_names: set[str] = set()
    for method in methods:
        metadata = getattr(method, "metadata", None)
        if metadata is None:
            raise TypeError(
                f"Discovery method {getattr(method, 'name', method)!r} is missing metadata"
            )
        if not isinstance(metadata, MethodMetadata):
            raise TypeError(
                f"Discovery method {getattr(method, 'name', method)!r} metadata must be "
                "MethodMetadata"
            )
        if metadata.name != method.name:
            raise ValueError(
                f"Discovery method name {method.name!r} does not match metadata "
                f"name {metadata.name!r}"
            )
        if metadata.name in seen_names:
            raise ValueError(f"Duplicate discovery method name: {metadata.name!r}")
        seen_names.add(metadata.name)


def _validate_finding_sets(
    finding_sets: Sequence[FindingSet],
    methods: Sequence[DiscoveryMethod],
) -> None:
    metadata_by_method = {method.name: method.metadata for method in methods}
    for finding_set in finding_sets:
        metadata = metadata_by_method.get(finding_set.method)
        if metadata is None:
            raise ValueError(
                f"FindingSet method {finding_set.method!r} does not match registered method"
            )
        if finding_set.method_version != metadata.version:
            raise ValueError(
                f"FindingSet version {finding_set.method_version!r} for "
                f"{finding_set.method!r} does not match metadata version "
                f"{metadata.version!r}"
            )


def _finding_sets_for_promotion(
    finding_sets: Sequence[FindingSet],
    methods: Sequence[DiscoveryMethod],
) -> list[FindingSet]:
    promotable_methods = {
        method.name
        for method in methods
        if method.metadata.plug_eligible
    }
    return [
        finding_set
        for finding_set in finding_sets
        if finding_set.method in promotable_methods
    ]
