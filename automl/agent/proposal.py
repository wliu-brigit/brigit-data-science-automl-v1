"""Proposal contract for the proposer-to-coder handoff."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping


DISALLOWED = ("parent_id",)


@dataclass(frozen=True)
class Proposal:
    schema_version: int
    slug: str
    strategy: str
    hypothesis: str
    implementation_plan: list[str]
    constraints: list[str]
    required_dependencies: list[str]
    rationale: str | None = None
    evidence: list[str] | None = None
    data_checks: list[str] | None = None
    risk_notes: str | None = None
    seed_hint: str | None = None
    required_preprocessing: list[dict] | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Proposal":
        names = set(proposal_field_names())
        return cls(**{name: payload[name] for name in names if name in payload})

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in proposal_field_names()}


def proposal_field_names() -> tuple[str, ...]:
    return tuple(field.name for field in fields(Proposal))


__all__ = ["DISALLOWED", "Proposal", "proposal_field_names"]
