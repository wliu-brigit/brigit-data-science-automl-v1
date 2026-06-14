"""Graph discovery adapters for the control-loop finding contract."""
from __future__ import annotations

from pathlib import Path

from projects.fraud_anomaly_detection.codex_poc.control.contract import Finding, FindingSet
from projects.fraud_anomaly_detection.codex_poc.control.discovery.metadata import (
    MethodMetadata,
)
from projects.fraud_anomaly_detection.graph.discover import residual_ring_members
from projects.fraud_anomaly_detection.graph.load import load_graph

METHOD_VERSION = "graph-skeleton-1"


class ResidualRingMethod:
    def __init__(self):
        self.metadata = MethodMetadata(
            name="graph:residual_ring_members",
            version=METHOD_VERSION,
            method_type="graph",
            time_semantics="snapshot_review",
            promotion_tier="review_queue",
            enforcement_projection="entity_key",
            params={"queue": "residual_ring_members"},
        )
        self.name = self.metadata.name

    def run(self, store: Path | str) -> FindingSet:
        graph = load_graph(store)
        queue = residual_ring_members(graph)
        findings = [
            Finding(
                user_id=str(row.user_id),
                score=float(row.ring_flagged),
                evidence={
                    "comp_id": int(row.comp_id),
                    "ring_users": int(row.ring_users),
                    "ring_types": int(row.ring_types),
                    "ring_flagged": int(row.ring_flagged),
                    "entity_types": row.entity_types,
                },
            )
            for row in queue.itertuples(index=False)
        ]
        return FindingSet(method=self.name, method_version=METHOD_VERSION, findings=findings)
