"""Typed task/problem declarations for project recipes."""

from __future__ import annotations

from dataclasses import dataclass


def _require_non_empty(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string, got {value!r}")


@dataclass(frozen=True)
class BinaryClassification:
    """Binary-classification problem definition."""

    target: str
    positive_label: object = 1

    def __post_init__(self) -> None:
        _require_non_empty(self.target, "target")


@dataclass(frozen=True)
class Regression:
    """Regression problem definition."""

    target: str

    def __post_init__(self) -> None:
        _require_non_empty(self.target, "target")


@dataclass(frozen=True)
class Multiclass:
    """Multiclass-classification problem definition."""

    target: str
    classes: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.target, "target")
        if self.classes is None:
            return
        if len(self.classes) == 0:
            raise ValueError("classes must be a non-empty tuple when provided")
        if len(set(self.classes)) != len(self.classes):
            raise ValueError(f"duplicate class label(s) in {self.classes!r}")


Task = BinaryClassification | Regression | Multiclass


__all__ = ["BinaryClassification", "Multiclass", "Regression", "Task"]
