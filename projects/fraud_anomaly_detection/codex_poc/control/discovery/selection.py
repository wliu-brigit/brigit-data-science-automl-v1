"""Reusable discovery-candidate selection for promotion into plug derivation."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from projects.fraud_anomaly_detection.codex_poc.control.discovery.metadata import (
    MethodMetadata,
)

OutcomeFn = Callable[[set[str]], dict]


@dataclass(frozen=True)
class DiscoveryCandidate:
    name: str
    users: set[str]
    metadata: MethodMetadata


@dataclass(frozen=True)
class SelectionRule:
    min_marginal_users: int = 10
    min_marginal_dpd45_user_rate: float = 0.50
    promotable_tiers: tuple[str, ...] = ("plug_candidate",)


@dataclass(frozen=True)
class SelectionRow:
    name: str
    users: set[str]
    total: dict
    net_new_users: set[str]
    net: dict
    marginal_users: set[str]
    marginal: dict
    selected: bool
    reason: str
    metadata: MethodMetadata


@dataclass(frozen=True)
class SelectionResult:
    selected: list[SelectionRow]
    excluded: list[SelectionRow]
    final_users: set[str]


def select_candidates(
    candidates: Sequence[DiscoveryCandidate],
    *,
    baseline_users: set[str],
    outcome_fn: OutcomeFn,
    rule: SelectionRule,
) -> SelectionResult:
    """Select candidates by marginal contribution after baseline and prior selections."""
    baseline = {str(user_id) for user_id in baseline_users}
    enriched = []
    for candidate in candidates:
        users = {str(user_id) for user_id in candidate.users}
        net_new_users = users - baseline
        enriched.append(
            {
                "candidate": candidate,
                "users": users,
                "total": outcome_fn(users),
                "net_new_users": net_new_users,
                "net": outcome_fn(net_new_users),
            }
        )

    enriched.sort(
        key=lambda item: (
            item["net"].get("dpd45_user_rate", 0.0),
            item["net"].get("users", 0),
            item["candidate"].name,
        ),
        reverse=True,
    )

    selected: list[SelectionRow] = []
    excluded: list[SelectionRow] = []
    covered = set(baseline)
    for item in enriched:
        candidate = item["candidate"]
        users = item["users"]
        marginal_users = users - covered
        marginal = outcome_fn(marginal_users)
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
    return SelectionResult(selected=selected, excluded=excluded, final_users=covered)


def _exclusion_reason(
    candidate: DiscoveryCandidate,
    marginal: dict,
    rule: SelectionRule,
) -> str:
    if candidate.metadata.promotion_tier not in rule.promotable_tiers:
        return "promotion_tier"
    if marginal.get("users", 0) < rule.min_marginal_users:
        return "min_marginal_users"
    if marginal.get("dpd45_user_rate", 0.0) < rule.min_marginal_dpd45_user_rate:
        return "min_marginal_dpd45_user_rate"
    return "selected"
