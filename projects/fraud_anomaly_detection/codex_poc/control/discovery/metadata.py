"""Semantic metadata for discovery methods in the fraud-control loop."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

MethodType = Literal["scenario", "graph", "model", "subgroup"]
TimeSemantics = Literal["snapshot_review", "leakfree_asof", "production_safe"]
PromotionTier = Literal["evidence_only", "review_queue", "plug_candidate"]
EnforcementProjection = Literal["entity_key", "scenario_rule", "none"]


@dataclass(frozen=True)
class MethodMetadata:
    """Reproducible semantics for a discovery method."""

    name: str
    version: str
    method_type: MethodType
    time_semantics: TimeSemantics
    promotion_tier: PromotionTier
    enforcement_projection: EnforcementProjection
    enabled: bool = True
    params: dict[str, object] = field(default_factory=dict)
