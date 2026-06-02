"""Evaluation result value objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd


def _tuple_of_dicts(value: object) -> tuple[dict[str, Any], ...]:
    if not value:
        return ()
    if isinstance(value, Mapping):
        return tuple({"name": str(key), "value": inner, "augmentations": []} for key, inner in value.items())
    return tuple(dict(item) for item in value)  # type: ignore[arg-type]


def _json_list(value: tuple[Any, ...]) -> list[Any]:
    return [dict(item) if isinstance(item, Mapping) else item for item in value]


@dataclass(frozen=True)
class EvalResult:
    label: str
    eval_dataset_id: str
    eval_dataset_kind: str
    predictions_uri: str
    predictions_manifest_uri: str
    augmentations_used: tuple[Any, ...]
    primary: str
    metrics: tuple[Mapping[str, Any], ...]
    computed_at: str
    schema_version: int = 1
    cached: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "augmentations_used", tuple(self.augmentations_used))
        object.__setattr__(self, "metrics", _tuple_of_dicts(self.metrics))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvalResult":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            label=str(payload["label"]),
            eval_dataset_id=str(payload["eval_dataset_id"]),
            eval_dataset_kind=str(payload["eval_dataset_kind"]),
            predictions_uri=str(payload.get("predictions_uri", "")),
            predictions_manifest_uri=str(payload.get("predictions_manifest_uri", "")),
            augmentations_used=tuple(payload.get("augmentations_used", ())),
            primary=str(payload["primary"]),
            metrics=_tuple_of_dicts(payload.get("metrics", ())),
            computed_at=str(payload["computed_at"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "label": self.label,
            "eval_dataset_id": self.eval_dataset_id,
            "eval_dataset_kind": self.eval_dataset_kind,
            "predictions_uri": self.predictions_uri,
            "predictions_manifest_uri": self.predictions_manifest_uri,
            "augmentations_used": _json_list(self.augmentations_used),
            "primary": self.primary,
            "metrics": [dict(record) for record in self.metrics],
            "computed_at": self.computed_at,
        }


@dataclass(frozen=True)
class EvalIndexEntry:
    label: str
    eval_dataset_id: str
    kind: str
    report_path: str
    eval_dataset_manifest_uri: str
    predictions_uri: str
    predictions_manifest_uri: str
    augmentations_used: tuple[Any, ...]
    computed_at: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvalIndexEntry":
        return cls(
            label=str(payload["label"]),
            eval_dataset_id=str(payload["eval_dataset_id"]),
            kind=str(payload["kind"]),
            report_path=str(payload["report_path"]),
            eval_dataset_manifest_uri=str(payload["eval_dataset_manifest_uri"]),
            predictions_uri=str(payload.get("predictions_uri", "")),
            predictions_manifest_uri=str(payload.get("predictions_manifest_uri", "")),
            augmentations_used=tuple(payload.get("augmentations_used", ())),
            computed_at=str(payload["computed_at"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "eval_dataset_id": self.eval_dataset_id,
            "kind": self.kind,
            "report_path": self.report_path,
            "eval_dataset_manifest_uri": self.eval_dataset_manifest_uri,
            "predictions_uri": self.predictions_uri,
            "predictions_manifest_uri": self.predictions_manifest_uri,
            "augmentations_used": _json_list(self.augmentations_used),
            "computed_at": self.computed_at,
        }


@dataclass(frozen=True)
class EvalIndex:
    primary_label: str | None
    evaluations: tuple[EvalIndexEntry, ...] = ()
    schema_version: int = 1

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvalIndex":
        primary = payload.get("primary_label")
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            primary_label=str(primary) if primary is not None else None,
            evaluations=tuple(
                EvalIndexEntry.from_dict(item) for item in payload.get("evaluations", ())
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "primary_label": self.primary_label,
            "evaluations": [entry.to_dict() for entry in self.evaluations],
        }


@dataclass(frozen=True)
class Predictions:
    trial_run_id: str
    eval_dataset_id: str
    eval_dataset_kind: str
    label: str
    hash_key: tuple[str, ...]
    frame: pd.DataFrame
    augmentations_used: tuple[Any, ...]
    written_at: str
    schema_version: int = 1

    def manifest_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trial_run_id": self.trial_run_id,
            "eval_dataset_id": self.eval_dataset_id,
            "eval_dataset_kind": self.eval_dataset_kind,
            "label": self.label,
            "hash_key": list(self.hash_key),
            "row_count": int(len(self.frame)),
            "augmentations_used": _json_list(self.augmentations_used),
            "written_at": self.written_at,
        }

    @classmethod
    def from_parts(cls, manifest: Mapping[str, Any], frame: pd.DataFrame) -> "Predictions":
        return cls(
            schema_version=int(manifest.get("schema_version", 1)),
            trial_run_id=str(manifest["trial_run_id"]),
            eval_dataset_id=str(manifest["eval_dataset_id"]),
            eval_dataset_kind=str(manifest["eval_dataset_kind"]),
            label=str(manifest["label"]),
            hash_key=tuple(str(item) for item in manifest.get("hash_key", ())),
            frame=frame,
            augmentations_used=tuple(manifest.get("augmentations_used", ())),
            written_at=str(manifest["written_at"]),
        )


__all__ = ["EvalIndex", "EvalIndexEntry", "EvalResult", "Predictions"]
