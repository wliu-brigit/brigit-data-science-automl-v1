"""Semantic metadata for discovery methods in the fraud-control loop."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

MethodType = Literal["scenario", "graph", "model", "subgroup"]
TimeSemantics = Literal["snapshot_review", "leakfree_asof", "production_safe"]
PromotionTier = Literal["evidence_only", "review_queue", "plug_candidate"]
EnforcementProjection = Literal["entity_key", "scenario_rule", "none"]
# Where the method is computed. GDS is flagged non-deterministic (approximate,
# order-dependent) and slower; Cypher and DuckDB are deterministic.
Source = Literal["duckdb", "neo4j_cypher", "neo4j_gds"]

VALID_METHOD_TYPES = frozenset(("scenario", "graph", "model", "subgroup"))
VALID_TIME_SEMANTICS = frozenset(("snapshot_review", "leakfree_asof", "production_safe"))
VALID_PROMOTION_TIERS = frozenset(("evidence_only", "review_queue", "plug_candidate"))
VALID_ENFORCEMENT_PROJECTIONS = frozenset(("entity_key", "scenario_rule", "none"))
VALID_SOURCES = frozenset(("duckdb", "neo4j_cypher", "neo4j_gds"))
PLUG_SAFE_TIME_SEMANTICS = frozenset(("leakfree_asof", "production_safe"))
NON_DETERMINISTIC_SOURCES = frozenset(("neo4j_gds",))
_SOURCE_LABELS = {
    "duckdb": "DuckDB",
    "neo4j_cypher": "Neo4j (Cypher)",
    "neo4j_gds": "Neo4j (GDS, non-deterministic)",
}


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
    params: Mapping[str, object] = field(default_factory=dict)
    # `source`: which engine computes this method (declared, not resolved at runtime).
    # `depends_on`: declared dependency tags, e.g. "column:is_fraud",
    # "scenario:ring_device_burst"; empty means structural (no seed).
    source: Source = "duckdb"
    depends_on: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must be non-empty")
        if not self.version.strip():
            raise ValueError("version must be non-empty")
        _require_member("method_type", self.method_type, VALID_METHOD_TYPES)
        _require_member("time_semantics", self.time_semantics, VALID_TIME_SEMANTICS)
        _require_member("promotion_tier", self.promotion_tier, VALID_PROMOTION_TIERS)
        _require_member(
            "enforcement_projection",
            self.enforcement_projection,
            VALID_ENFORCEMENT_PROJECTIONS,
        )
        _require_member("source", self.source, VALID_SOURCES)
        object.__setattr__(self, "depends_on", tuple(self.depends_on))
        if self.promotion_tier == "plug_candidate" and self.enforcement_projection == "none":
            raise ValueError("plug_candidate methods must declare an enforcement projection")
        if (
            self.promotion_tier == "plug_candidate"
            and self.time_semantics not in PLUG_SAFE_TIME_SEMANTICS
        ):
            raise ValueError(
                "plug_candidate methods must be leakfree_asof or production_safe"
            )
        object.__setattr__(self, "params", _freeze_value(self.params))

    @property
    def plug_eligible(self) -> bool:
        return (
            self.promotion_tier == "plug_candidate"
            and self.enforcement_projection != "none"
            and self.time_semantics in PLUG_SAFE_TIME_SEMANTICS
        )

    @property
    def deterministic(self) -> bool:
        return self.source not in NON_DETERMINISTIC_SOURCES

    @property
    def source_label(self) -> str:
        return source_label(self.source)

    @property
    def depends_on_label(self) -> str:
        return depends_on_label(self.depends_on)


def source_label(source: str) -> str:
    """Human-readable engine label, e.g. 'Neo4j (GDS, non-deterministic)'."""
    return _SOURCE_LABELS[source]


def depends_on_label(depends_on: tuple[str, ...]) -> str:
    """Render declared dependency tags; 'structural' when there are none."""
    return "; ".join(depends_on) if depends_on else "structural"


def _require_member(name: str, value: str, allowed: frozenset[str]) -> None:
    if value not in allowed:
        allowed_values = ", ".join(sorted(allowed))
        raise ValueError(f"{name} must be one of {allowed_values}; got {value!r}")


def _freeze_value(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_freeze_value(item) for item in value)
    return value
