"""Project overview domain value objects."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectOverview:
    schema_version: int = 1
    project_name: str = ""
    created_at: str = ""
    current_experiment_id: str | None = None
    dataset_count: int = 0


__all__ = ["ProjectOverview"]
