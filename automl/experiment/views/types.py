"""Typed experiment view results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from automl.trial.types import TrialDetails, TrialSummary


@dataclass(frozen=True)
class LeaderboardData:
    schema_version: int = 1
    metric: str = ""
    experiment_id: str = ""
    rows: tuple[TrialSummary, ...] = ()
    n_unscored: int = 0

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LeaderboardData":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            metric=str(payload.get("metric", "")),
            experiment_id=str(payload.get("experiment_id", "")),
            rows=tuple(TrialSummary.from_dict(item) for item in payload.get("rows", ())),
            n_unscored=int(payload.get("n_unscored", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "metric": self.metric,
            "experiment_id": self.experiment_id,
            "rows": [row.to_dict() for row in self.rows],
            "n_unscored": self.n_unscored,
        }


@dataclass(frozen=True)
class MetricDelta:
    metric: str = ""
    value_a: float | None = None
    value_b: float | None = None
    delta: float | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MetricDelta":
        return cls(
            metric=str(payload.get("metric", "")),
            value_a=_optional_float(payload.get("value_a")),
            value_b=_optional_float(payload.get("value_b")),
            delta=_optional_float(payload.get("delta")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "value_a": self.value_a,
            "value_b": self.value_b,
            "delta": self.delta,
        }


@dataclass(frozen=True)
class ComparisonResult:
    schema_version: int = 1
    run_ids: tuple[str, ...] = ()
    runs: tuple[TrialDetails, ...] = ()
    metric_deltas: tuple[MetricDelta, ...] = ()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ComparisonResult":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            run_ids=tuple(str(item) for item in payload.get("run_ids", ())),
            runs=tuple(TrialDetails.from_dict(item) for item in payload.get("runs", ())),
            metric_deltas=tuple(
                MetricDelta.from_dict(item) for item in payload.get("metric_deltas", ())
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_ids": list(self.run_ids),
            "runs": [run.to_dict() for run in self.runs],
            "metric_deltas": [delta.to_dict() for delta in self.metric_deltas],
        }


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)  # type: ignore[arg-type]


__all__ = ["ComparisonResult", "LeaderboardData", "MetricDelta"]
