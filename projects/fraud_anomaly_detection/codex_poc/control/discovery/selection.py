"""Reusable discovery-candidate selection for promotion into plug derivation."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from projects.fraud_anomaly_detection.codex_poc.control.discovery.metadata import (
    MethodMetadata,
)

Outcome = Mapping[str, int | float]
OutcomeFn = Callable[[frozenset[str]], Outcome]
_REQUIRED_OUTCOME_KEYS = ("users", "dpd45_user_rate")


@dataclass(frozen=True)
class DiscoveryCandidate:
    name: str
    users: frozenset[str]
    metadata: MethodMetadata

    def __init__(
        self,
        name: str,
        users: set[str] | frozenset[str],
        metadata: MethodMetadata,
    ):
        if name != metadata.name:
            raise ValueError(
                f"DiscoveryCandidate name {name!r} must match metadata.name "
                f"{metadata.name!r}"
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "users", frozenset(str(user_id) for user_id in users))
        object.__setattr__(self, "metadata", metadata)


@dataclass(frozen=True)
class SelectionRule:
    min_marginal_users: int = 10
    min_marginal_dpd45_user_rate: float = 0.50
    promotable_tiers: tuple[str, ...] = ("plug_candidate",)


@dataclass(frozen=True)
class SelectionRow:
    name: str
    users: frozenset[str]
    total: Outcome
    net_new_users: frozenset[str]
    net: Outcome
    marginal_users: frozenset[str]
    marginal: Outcome
    selected: bool
    reason: str
    metadata: MethodMetadata


@dataclass(frozen=True)
class SelectionResult:
    selected: list[SelectionRow]
    excluded: list[SelectionRow]
    final_users: frozenset[str]


def select_candidates(
    candidates: Sequence[DiscoveryCandidate],
    *,
    baseline_users: set[str],
    outcome_fn: OutcomeFn,
    rule: SelectionRule,
) -> SelectionResult:
    """Select candidates by marginal contribution after baseline and prior selections."""
    baseline = frozenset(str(user_id) for user_id in baseline_users)
    enriched = []
    for order, candidate in enumerate(candidates):
        users = frozenset(str(user_id) for user_id in candidate.users)
        net_new_users = frozenset(users - baseline)
        enriched.append(
            {
                "candidate": candidate,
                "users": users,
                "total": _freeze_outcome(outcome_fn(users)),
                "net_new_users": net_new_users,
                "net": _freeze_outcome(outcome_fn(net_new_users)),
                "order": order,
            }
        )

    enriched.sort(
        key=lambda item: (
            -_metric(item["net"], "dpd45_user_rate"),
            -_metric(item["net"], "users"),
            item["order"],
        )
    )

    selected: list[SelectionRow] = []
    excluded: list[SelectionRow] = []
    covered = set(baseline)
    for item in enriched:
        candidate = item["candidate"]
        users = item["users"]
        marginal_users = frozenset(users - covered)
        marginal = _freeze_outcome(outcome_fn(marginal_users))
        reason = _exclusion_reason(candidate, marginal, rule)
        include = reason == "selected"
        row = SelectionRow(
            name=candidate.name,
            users=users,
            total=item["total"],
            net_new_users=item["net_new_users"],
            net=item["net"],
            marginal_users=marginal_users,
            marginal=marginal,
            selected=include,
            reason=reason,
            metadata=candidate.metadata,
        )
        if include:
            selected.append(row)
            covered |= users
        else:
            excluded.append(row)
    return SelectionResult(selected=selected, excluded=excluded, final_users=frozenset(covered))


def _exclusion_reason(
    candidate: DiscoveryCandidate,
    marginal: dict,
    rule: SelectionRule,
) -> str:
    if candidate.metadata.promotion_tier not in rule.promotable_tiers:
        return "promotion_tier"
    if _metric(marginal, "users") < rule.min_marginal_users:
        return "min_marginal_users"
    if _metric(marginal, "dpd45_user_rate") < rule.min_marginal_dpd45_user_rate:
        return "min_marginal_dpd45_user_rate"
    return "selected"


def _freeze_outcome(outcome: Outcome) -> Outcome:
    missing = [key for key in _REQUIRED_OUTCOME_KEYS if key not in outcome]
    if missing:
        raise KeyError(f"Outcome is missing required keys: {', '.join(missing)}")
    return MappingProxyType(dict(outcome))


def _metric(outcome: Outcome, key: str) -> int | float:
    return outcome[key]
