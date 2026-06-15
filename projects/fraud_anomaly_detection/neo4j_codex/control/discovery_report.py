"""Discovery-method backtest summaries over scenario and graph findings."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from projects.fraud_anomaly_detection.neo4j_codex.control.contract import FindingSet
from projects.fraud_anomaly_detection.neo4j_codex.control.discovery.metadata import (
    MethodMetadata,
)
from projects.fraud_anomaly_detection.neo4j_codex.control.outcomes import summarize_users


def summarize_discovery(
    store: Path | str,
    finding_sets: Sequence[FindingSet],
    method_metadata: Sequence[MethodMetadata],
    eligible_users: Iterable[str] | None = None,
    start_ts: pd.Timestamp | None = None,
    end_ts: pd.Timestamp | None = None,
) -> dict:
    """Report method-level outcomes, union outcomes, and scenario/graph overlap."""
    _validate_metadata(finding_sets, method_metadata)
    metadata_by_method = {metadata.name: metadata for metadata in method_metadata}
    eligible = None if eligible_users is None else {str(user_id) for user_id in eligible_users}
    method_reports = []
    union_users: set[str] = set()
    users_by_type: dict[str, set[str]] = {
        metadata.method_type: set() for metadata in method_metadata
    }

    for finding_set in finding_sets:
        metadata = metadata_by_method[finding_set.method]
        frame = finding_set.to_frame()
        if eligible is not None and not frame.empty:
            frame = frame[frame["user_id"].isin(eligible)]

        method_users = set(frame["user_id"].astype(str)) if not frame.empty else set()
        union_users.update(method_users)
        users_by_type[metadata.method_type].update(method_users)

        method_reports.append(
            {
                "method": finding_set.method,
                "method_version": finding_set.method_version,
                "method_type": metadata.method_type,
                "time_semantics": metadata.time_semantics,
                "promotion_tier": metadata.promotion_tier,
                "enforcement_projection": metadata.enforcement_projection,
                "findings": int(len(frame)),
                "n_users": len(method_users),
                "outcomes": summarize_users(store, method_users, start_ts=start_ts, end_ts=end_ts),
            }
        )

    return {
        "methods": method_reports,
        "union": {
            "n_users": len(union_users),
            "outcomes": summarize_users(store, union_users, start_ts=start_ts, end_ts=end_ts),
        },
        "attribution": {
            "by_method_type": {
                method_type: len(users) for method_type, users in sorted(users_by_type.items())
            },
            "multi_type_users": _multi_type_user_count(users_by_type),
        },
    }


def _validate_metadata(
    finding_sets: Sequence[FindingSet],
    method_metadata: Sequence[MethodMetadata],
) -> None:
    if len(finding_sets) != len(method_metadata):
        raise ValueError(
            f"Expected one metadata record per FindingSet; got {len(method_metadata)} "
            f"metadata records for {len(finding_sets)} finding sets"
        )
    for finding_set, metadata in zip(finding_sets, method_metadata, strict=True):
        if finding_set.method != metadata.name:
            raise ValueError(
                f"FindingSet method {finding_set.method!r} does not match metadata "
                f"{metadata.name!r}"
            )
        if finding_set.method_version != metadata.version:
            raise ValueError(
                f"FindingSet version {finding_set.method_version!r} does not match "
                f"metadata version {metadata.version!r}"
            )


def _multi_type_user_count(users_by_type: dict[str, set[str]]) -> int:
    seen_types_by_user: dict[str, set[str]] = {}
    for method_type, users in users_by_type.items():
        for user_id in users:
            seen_types_by_user.setdefault(user_id, set()).add(method_type)
    return sum(1 for method_types in seen_types_by_user.values() if len(method_types) > 1)
