"""Trial read-model value objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from automl.eval.results import EvalResult


class TrialStatus(str, Enum):
    UNKNOWN = "UNKNOWN"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    KILLED = "KILLED"


@dataclass(frozen=True)
class ArtifactRef:
    path: str = ""
    file_size: int | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactRef":
        return cls(
            path=str(payload.get("path", "")),
            file_size=_optional_int(payload.get("file_size")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "file_size": self.file_size}


@dataclass(frozen=True)
class TrialSummary:
    schema_version: int = 1
    run_id: str = ""
    slug: str = ""
    strategy: str = ""
    status: TrialStatus = TrialStatus.UNKNOWN
    primary_metric_name: str = ""
    primary_metric_value: float | None = None
    started_at: str | None = None
    ended_at: str | None = None
    parent_run_id: str | None = None
    dataset_hash: str | None = None
    trial_number: int | None = None
    hypothesis: str = ""
    training_origin: str = ""
    training_time_s: float | None = None
    n_features: int | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrialSummary":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            run_id=str(payload.get("run_id", "")),
            slug=str(payload.get("slug", "")),
            strategy=str(payload.get("strategy", "")),
            status=_status(payload.get("status")),
            primary_metric_name=str(payload.get("primary_metric_name", "")),
            primary_metric_value=_optional_float(payload.get("primary_metric_value")),
            started_at=_optional_str(payload.get("started_at")),
            ended_at=_optional_str(payload.get("ended_at")),
            parent_run_id=_optional_str(payload.get("parent_run_id")),
            dataset_hash=_optional_str(payload.get("dataset_hash")),
            trial_number=_optional_int(payload.get("trial_number")),
            hypothesis=str(payload.get("hypothesis", "")),
            training_origin=str(payload.get("training_origin", "")),
            training_time_s=_optional_float(payload.get("training_time_s")),
            n_features=_optional_int(payload.get("n_features")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "slug": self.slug,
            "strategy": self.strategy,
            "status": self.status.value,
            "primary_metric_name": self.primary_metric_name,
            "primary_metric_value": self.primary_metric_value,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "parent_run_id": self.parent_run_id,
            "dataset_hash": self.dataset_hash,
            "trial_number": self.trial_number,
            "hypothesis": self.hypothesis,
            "training_origin": self.training_origin,
            "training_time_s": self.training_time_s,
            "n_features": self.n_features,
        }


@dataclass(frozen=True)
class TrialDetails:
    run_id: str = ""
    status: TrialStatus = TrialStatus.UNKNOWN
    params: Mapping[str, str] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)
    tags: Mapping[str, str] = field(default_factory=dict)
    artifacts: tuple[ArtifactRef, ...] = ()
    evaluations: tuple["EvalResult", ...] | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", dict(self.params))
        object.__setattr__(self, "metrics", dict(self.metrics))
        object.__setattr__(self, "tags", dict(self.tags))
        object.__setattr__(
            self,
            "artifacts",
            tuple(
                item if isinstance(item, ArtifactRef) else ArtifactRef.from_dict(item)
                for item in self.artifacts
            ),
        )
        if self.evaluations is not None:
            eval_result_type = _eval_result_type()
            object.__setattr__(
                self,
                "evaluations",
                tuple(
                    item if isinstance(item, eval_result_type) else eval_result_type.from_dict(item)
                    for item in self.evaluations
                ),
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrialDetails":
        evaluations = payload.get("evaluations", None)
        eval_result_type = _eval_result_type()
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            run_id=str(payload.get("run_id", "")),
            status=_status(payload.get("status")),
            params=dict(payload.get("params", {})),
            metrics={str(key): float(value) for key, value in dict(payload.get("metrics", {})).items()},
            tags={str(key): str(value) for key, value in dict(payload.get("tags", {})).items()},
            artifacts=tuple(ArtifactRef.from_dict(item) for item in payload.get("artifacts", ())),
            evaluations=None
            if evaluations is None
            else tuple(eval_result_type.from_dict(item) for item in evaluations),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "status": self.status.value,
            "params": dict(self.params),
            "metrics": dict(self.metrics),
            "tags": dict(self.tags),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "evaluations": None
            if self.evaluations is None
            else [evaluation.to_dict() for evaluation in self.evaluations],
        }


@dataclass(frozen=True)
class ParentExperimentRef:
    schema_version: int = 1
    mlflow_experiment_id: str = ""
    mlflow_experiment_name: str = ""
    dry_run: bool = False
    project_name: str = ""
    experiment_id: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ParentExperimentRef":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            mlflow_experiment_id=str(payload.get("mlflow_experiment_id", "")),
            mlflow_experiment_name=str(payload.get("mlflow_experiment_name", "")),
            dry_run=bool(payload.get("dry_run", False)),
            project_name=str(payload.get("project_name", "")),
            experiment_id=str(payload.get("experiment_id", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mlflow_experiment_id": self.mlflow_experiment_id,
            "mlflow_experiment_name": self.mlflow_experiment_name,
            "dry_run": self.dry_run,
            "project_name": self.project_name,
            "experiment_id": self.experiment_id,
        }


def _status(value: object) -> TrialStatus:
    if isinstance(value, TrialStatus):
        return value
    try:
        return TrialStatus(str(value or "").upper())
    except ValueError:
        return TrialStatus.UNKNOWN


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    return int(value)  # type: ignore[arg-type]


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)  # type: ignore[arg-type]


def _eval_result_type():
    from automl.eval.results import EvalResult

    return EvalResult


__all__ = [
    "ArtifactRef",
    "ParentExperimentRef",
    "TrialDetails",
    "TrialStatus",
    "TrialSummary",
]
