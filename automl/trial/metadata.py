"""Write-side trial schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping
import json


@dataclass(frozen=True)
class ModelSource:
    source: str = ""
    artifact_path: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "ModelSource | None":
        if payload is None:
            return None
        return cls(
            source=str(payload.get("source", "")),
            artifact_path=str(payload.get("artifact_path", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SeedSelection:
    schema_version: int = 1
    selector: str = ""
    run_id: str = ""
    trial_id: str = ""
    metric_name: str = ""
    metric_value: float | None = None
    strategy: str = ""
    model_source: ModelSource | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "SeedSelection | None":
        if payload is None:
            return None
        metric_value = payload.get("metric_value")
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            selector=str(payload.get("selector", "")),
            run_id=str(payload.get("run_id", "")),
            trial_id=str(payload.get("trial_id", "")),
            metric_name=str(payload.get("metric_name", "")),
            metric_value=None if metric_value in (None, "") else float(metric_value),
            strategy=str(payload.get("strategy", "")),
            model_source=ModelSource.from_dict(_mapping_or_none(payload.get("model_source"))),
        )

    def to_dict(self) -> dict[str, Any]:
        document = asdict(self)
        if self.model_source is None:
            document["model_source"] = None
        return document


@dataclass(frozen=True)
class TrialMetadata:
    schema_version: int = 1
    slug: str = ""
    strategy: str = ""
    hypothesis: str = ""
    training_origin: str = ""
    created_at: str = ""
    project_name: str = ""
    project_package: str = ""
    experiment_id: str = ""
    seed: SeedSelection | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrialMetadata":
        seed = SeedSelection.from_dict(_mapping_or_none(payload.get("seed")))
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            slug=str(payload.get("slug", "")),
            strategy=str(payload.get("strategy", "")),
            hypothesis=str(payload.get("hypothesis", "")),
            training_origin=str(payload.get("training_origin", "")),
            created_at=str(payload.get("created_at", "")),
            project_name=str(payload.get("project_name", "")),
            project_package=str(payload.get("project_package", "")),
            experiment_id=str(payload.get("experiment_id", "")),
            seed=seed,
        )

    @classmethod
    def read(cls, path: str | Path) -> "TrialMetadata":
        with Path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError("trial metadata must be a JSON object")
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        document = asdict(self)
        document["seed"] = self.seed.to_dict() if self.seed is not None else None
        return document

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target


@dataclass(frozen=True)
class TimingReport:
    schema_version: int = 2
    unit: str = "seconds"
    total_seconds: float = 0.0
    phases: dict[str, float] | None = None
    phase_details: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TimingReport":
        phases = payload.get("phases") or {}
        if not isinstance(phases, Mapping):
            raise ValueError("timing phases must be a mapping")
        details = payload.get("phase_details") or {}
        if details and not isinstance(details, Mapping):
            raise ValueError("timing phase_details must be a mapping")
        return cls(
            schema_version=int(payload.get("schema_version", 2)),
            unit=str(payload.get("unit", "seconds")),
            total_seconds=float(payload.get("total_seconds", 0.0)),
            phases={str(key): float(value) for key, value in phases.items()},
            phase_details={
                str(key): dict(value)
                for key, value in details.items()
                if isinstance(value, Mapping)
            }
            if details
            else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mapping_or_none(value: Any) -> Mapping[str, Any] | None:
    if value in (None, ""):
        return None
    if is_dataclass(value):
        return asdict(value)
    if not isinstance(value, Mapping):
        raise ValueError(f"expected mapping, got {type(value).__name__}")
    return value


__all__ = [
    "ModelSource",
    "SeedSelection",
    "TimingReport",
    "TrialMetadata",
]
