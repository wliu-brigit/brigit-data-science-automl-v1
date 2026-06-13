"""Discovery-method backtest summaries over scenario and graph findings."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from projects.fraud_anomaly_detection.codex_poc.control.contract import FindingSet
from projects.fraud_anomaly_detection.codex_poc.control.outcomes import summarize_users


def summarize_discovery(
    store: Path | str,
    finding_sets: Sequence[FindingSet],
    eligible_users: Iterable[str] | None = None,
    start_ts: pd.Timestamp | None = None,
    end_ts: pd.Timestamp | None = None,
) -> dict:
    """Report method-level outcomes, union outcomes, and scenario/graph overlap."""
    eligible = None if eligible_users is None else {str(user_id) for user_id in eligible_users}
    method_reports = []
    union_users: set[str] = set()
    scenario_users: set[str] = set()
    graph_users: set[str] = set()

    for finding_set in finding_sets:
        frame = finding_set.to_frame()
        if eligible is not None and not frame.empty:
            frame = frame[frame["user_id"].isin(eligible)]

        method_users = set(frame["user_id"].astype(str)) if not frame.empty else set()
        union_users.update(method_users)
        if finding_set.method.startswith("scenario:"):
            scenario_users.update(method_users)
        elif finding_set.method.startswith("graph:"):
            graph_users.update(method_users)

        method_reports.append(
            {
                "method": finding_set.method,
                "method_version": finding_set.method_version,
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
            "scenario_only_users": len(scenario_users - graph_users),
            "graph_only_users": len(graph_users - scenario_users),
            "scenario_and_graph_users": len(scenario_users & graph_users),
        },
    }
