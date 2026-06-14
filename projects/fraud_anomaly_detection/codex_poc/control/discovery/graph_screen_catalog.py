"""Reviewed graph-screen descriptors for discovery reports and future live methods."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from projects.fraud_anomaly_detection.codex_poc.control.discovery.metadata import (
    MethodMetadata,
    PromotionTier,
)
from projects.fraud_anomaly_detection.codex_poc.control.discovery.selection import (
    DiscoveryCandidate,
)

GRAPH_SCREEN_VERSION = "selected-report-1"


@dataclass(frozen=True)
class GraphScreenSpec:
    """Metadata descriptor for a graph discovery screen."""

    name: str
    promotion_tier: PromotionTier
    params: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))

    @property
    def metadata(self) -> MethodMetadata:
        return MethodMetadata(
            name=f"graph:{self.name}",
            version=GRAPH_SCREEN_VERSION,
            method_type="graph",
            time_semantics="snapshot_review",
            promotion_tier=self.promotion_tier,
            enforcement_projection="entity_key",
            params={
                "display_name": self.name,
                "source": "selected_discovery_report",
                **self.params,
            },
        )

    def candidate(self, users: set[str]) -> DiscoveryCandidate:
        metadata = self.metadata
        return DiscoveryCandidate(
            name=metadata.name,
            users={str(user_id) for user_id in users},
            metadata=metadata,
        )


def default_graph_screen_specs(scenario_names: Iterable[str]) -> list[GraphScreenSpec]:
    """Return graph screens reviewed for the selected-discovery report."""
    specs = [
        GraphScreenSpec("residual_ring_members", promotion_tier="review_queue"),
        GraphScreenSpec("suspicion_queue_top200", promotion_tier="review_queue"),
        GraphScreenSpec("fraud_neighbours_hops2", promotion_tier="review_queue"),
        GraphScreenSpec(
            "high_risk_entity_members_scenario_fraud_seed",
            promotion_tier="plug_candidate",
        ),
        GraphScreenSpec(
            "multi_witness_neighbors_scenario_fraud_seed",
            promotion_tier="review_queue",
        ),
    ]
    specs.extend(
        GraphScreenSpec(
            f"scenario_neighborhood:{scenario_name}",
            promotion_tier="review_queue",
            params={"scenario_name": scenario_name},
        )
        for scenario_name in scenario_names
    )
    return specs
