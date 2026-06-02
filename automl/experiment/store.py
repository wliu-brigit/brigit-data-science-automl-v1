"""Experiment read-model value objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ExperimentOverview:
    schema_version: int = 1
    run_id: str = ""
    experiment_id: str = ""
    project_name: str = ""
    created_at: str = ""
    dry_run: bool = False

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperimentOverview":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            run_id=str(payload.get("run_id", "")),
            experiment_id=str(payload.get("experiment_id", "")),
            project_name=str(payload.get("project_name", "")),
            created_at=str(payload.get("created_at", "")),
            dry_run=bool(payload.get("dry_run", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "project_name": self.project_name,
            "created_at": self.created_at,
            "dry_run": self.dry_run,
        }


Experiment = ExperimentOverview


__all__ = ["Experiment", "ExperimentOverview"]
