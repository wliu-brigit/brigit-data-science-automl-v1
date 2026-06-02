"""Agent-domain validation checks."""

from __future__ import annotations

from dataclasses import MISSING, fields
from typing import Any

from automl.agent.proposal import DISALLOWED, Proposal, proposal_field_names
from automl.project.dependencies import allowed_dependencies
from automl.utils.slug import SLUG_RE
from automl.validate.base import Issue


def proposal_schema(proposal: dict[str, Any], *, session=None) -> list[Issue]:
    if not isinstance(proposal, dict):
        return [
            Issue(
                level="error",
                check="proposal.not_object",
                message="proposal must be a JSON object",
            )
        ]

    issues: list[Issue] = []
    for field_name in _required_field_names():
        if field_name not in proposal:
            issues.append(
                Issue(
                    level="error",
                    check="proposal.missing_field",
                    message=f"required field missing: {field_name}",
                    location=field_name,
                )
            )

    if type(proposal.get("schema_version")) is not int or proposal.get("schema_version") != 2:
        issues.append(
            Issue(
                level="error",
                check="proposal.bad_schema_version",
                message="schema_version must be 2",
                location="schema_version",
            )
        )

    slug = proposal.get("slug")
    if not isinstance(slug, str) or not SLUG_RE.fullmatch(slug):
        issues.append(
            Issue(
                level="error",
                check="proposal.bad_slug",
                message="slug must be lowercase snake_case and start with a letter",
                location="slug",
            )
        )

    for field_name in ("strategy", "hypothesis"):
        value = proposal.get(field_name)
        if not isinstance(value, str) or not value.strip():
            issues.append(
                Issue(
                    level="error",
                    check=f"proposal.bad_{field_name}",
                    message=f"{field_name} must be a non-empty string",
                    location=field_name,
                )
            )

    for field_name in ("implementation_plan", "constraints"):
        if not _non_empty_string_list(proposal.get(field_name)):
            issues.append(
                Issue(
                    level="error",
                    check=f"proposal.bad_{field_name}",
                    message=f"{field_name} must be a non-empty list of strings",
                    location=field_name,
                )
            )

    required_dependencies = proposal.get("required_dependencies")
    if not _non_empty_string_list(required_dependencies):
        issues.append(
            Issue(
                level="error",
                check="proposal.bad_required_dependencies",
                message="required_dependencies must be a non-empty list of strings",
                location="required_dependencies",
            )
        )
        required_dependencies = []

    seed_hint = proposal.get("seed_hint")
    if seed_hint is not None and (not isinstance(seed_hint, str) or not seed_hint.strip()):
        issues.append(
            Issue(
                level="error",
                check="proposal.bad_seed_hint",
                message="seed_hint must be a non-empty string when provided",
                location="seed_hint",
            )
        )
    elif isinstance(seed_hint, str) and not _valid_seed_hint(seed_hint):
        issues.append(
            Issue(
                level="error",
                check="proposal.bad_seed_hint",
                message="seed_hint must be 'auto', 'best', 'latest', or 'strategy:<name>'",
                location="seed_hint",
            )
        )

    for field_name in DISALLOWED:
        if field_name in proposal:
            issues.append(
                Issue(
                    level="error",
                    check=f"proposal.{field_name}_removed",
                    message=f"{field_name} is no longer accepted; seeds are resolved by metric query",
                    location=field_name,
                )
            )

    known_fields = set(proposal_field_names()) | set(DISALLOWED)
    for field_name in proposal:
        if field_name not in known_fields:
            issues.append(
                Issue(
                    level="warning",
                    check="proposal.unknown_field",
                    message=f"unknown field (will be ignored): {field_name}",
                    location=field_name,
                )
            )

    allowed = set(allowed_dependencies(session))
    for dependency in required_dependencies:
        if dependency not in allowed:
            issues.append(
                Issue(
                    level="error",
                    check="proposal.dep_not_allowed",
                    message=f"{dependency} not in allowed_dependencies",
                    location="required_dependencies",
                )
            )
    return issues


def _required_field_names() -> tuple[str, ...]:
    return tuple(
        field.name
        for field in fields(Proposal)
        if field.default is MISSING and field.default_factory is MISSING
    )


def _non_empty_string_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and item.strip() for item in value)
    )


def _valid_seed_hint(value: str) -> bool:
    raw = value.strip()
    return raw in {"auto", "best", "latest"} or (
        raw.startswith("strategy:") and bool(raw.split(":", 1)[1].strip())
    )


__all__ = ["proposal_schema"]
