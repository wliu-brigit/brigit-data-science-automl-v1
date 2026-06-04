"""Validate vocabulary: check-result value objects and the check runner.

This module is a leaf — it imports nothing from the rest of the library.
Domains define their own checks (functions yielding ``Issue``s) and their own
validation recipes (which checks to run, in what order), composing them with
``run_check`` into a ``ValidationReport``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class Issue:
    level: Severity
    check: str
    message: str
    location: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Issue":
        return cls(
            level=str(payload.get("level", "error")),  # type: ignore[arg-type]
            check=str(payload.get("check", "")),
            message=str(payload.get("message", "")),
            location=_optional_str(payload.get("location")),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "check": self.check,
            "message": self.message,
            "location": self.location,
        }


@dataclass(frozen=True)
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)
    schema_version: int = 1

    @property
    def passed(self) -> bool:
        return not any(issue.level == "error" for issue in self.issues)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ValidationReport":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            issues=[
                issue if isinstance(issue, Issue) else Issue.from_dict(issue)
                for issue in payload.get("issues", [])
            ],
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "passed": self.passed,
            "issues": [issue.to_json() for issue in self.issues],
        }


def run_check(name: str, fn: Callable[..., Iterable[Issue]], **kwargs) -> list[Issue]:
    """Run one check, reporting a crashed check as an error issue."""
    try:
        return list(fn(**kwargs))
    except Exception as exc:  # noqa: BLE001 - validation must report crashed checks
        return [
            Issue(
                level="error",
                check=f"{name}.crashed",
                message=f"check {name!r} raised {type(exc).__name__}: {exc}",
            )
        ]


def _optional_str(value: object) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


__all__ = ["Issue", "Severity", "ValidationReport", "run_check"]
