"""Reviewed graph-screen descriptors for discovery reports and future live methods."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from projects.fraud_anomaly_detection.neo4j_codex.control.discovery.metadata import (
    MethodMetadata,
    PromotionTier,
    Source,
)
from projects.fraud_anomaly_detection.neo4j_codex.control.discovery.selection import (
    DiscoveryCandidate,
)

GRAPH_SCREEN_VERSION = "control-loop-report-1"


@dataclass(frozen=True)
class GraphScreenSpec:
    """Metadata descriptor for a graph discovery screen."""

    name: str
    promotion_tier: PromotionTier
    source: Source = "neo4j_cypher"
    depends_on: tuple[str, ...] = ()
    params: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))
        object.__setattr__(self, "depends_on", tuple(self.depends_on))

    @property
    def metadata(self) -> MethodMetadata:
        return MethodMetadata(
            name=f"graph:{self.name}",
            version=GRAPH_SCREEN_VERSION,
            method_type="graph",
            time_semantics="snapshot_review",
            promotion_tier=self.promotion_tier,
            enforcement_projection="entity_key",
            source=self.source,
            depends_on=self.depends_on,
            params={
                "display_name": self.name,
                "catalog": "control_loop_report",
                **self.params,
            },
        )

    def candidate(self, users: Iterable[str]) -> DiscoveryCandidate:
        metadata = self.metadata
        return DiscoveryCandidate(
            name=metadata.name,
            users={str(user_id) for user_id in users},
            metadata=metadata,
        )


def default_graph_screen_specs(scenario_names: Iterable[str]) -> list[GraphScreenSpec]:
    """Return graph screens reviewed for the control-loop report."""
    specs = [
        GraphScreenSpec(
            "residual_ring_members",
            promotion_tier="review_queue",
            source="neo4j_gds",
            depends_on=("column:is_fraud",),
        ),
        GraphScreenSpec(
            "suspicion_queue_top200",
            promotion_tier="review_queue",
            source="neo4j_gds",
        ),
        GraphScreenSpec(
            "fraud_neighbours_hops2",
            promotion_tier="review_queue",
            source="neo4j_cypher",
            depends_on=("column:is_fraud",),
        ),
        GraphScreenSpec(
            "high_risk_entity_members_scenario_fraud_seed",
            promotion_tier="review_queue",
            source="neo4j_cypher",
            depends_on=("column:is_fraud", "scenario:any"),
        ),
        GraphScreenSpec(
            "multi_witness_neighbors_scenario_fraud_seed",
            promotion_tier="review_queue",
            source="neo4j_cypher",
            depends_on=("column:is_fraud", "scenario:any"),
        ),
    ]
    specs.extend(
        GraphScreenSpec(
            f"scenario_neighborhood:{scenario_name}",
            promotion_tier="review_queue",
            source="neo4j_cypher",
            depends_on=(f"scenario:{scenario_name}",),
            params={"scenario_name": scenario_name},
        )
        for scenario_name in scenario_names
    )
    _require_unique_specs(specs)
    return specs


def _require_unique_specs(specs: Iterable[GraphScreenSpec]) -> None:
    seen_names: set[str] = set()
    seen_metadata_names: set[str] = set()
    for spec in specs:
        if spec.name in seen_names:
            raise ValueError(f"Duplicate graph screen name: {spec.name!r}")
        seen_names.add(spec.name)
        metadata_name = spec.metadata.name
        if metadata_name in seen_metadata_names:
            raise ValueError(f"Duplicate graph screen metadata name: {metadata_name!r}")
        seen_metadata_names.add(metadata_name)
