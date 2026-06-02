"""Typed schema for the runner's trial manifest artifact."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class TrialRunManifest:
    schema_version: int
    run: dict[str, Any]
    data: dict[str, Any]
    model: dict[str, Any]
    evaluation: dict[str, Any]
    validation: dict[str, Any]
    timing: dict[str, Any]
    artifacts: tuple[dict[str, Any], ...]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrialRunManifest":
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, list):
            raise ValueError("trial run manifest artifacts must be a list")
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            run=_dict_value(payload, "run"),
            data=_dict_value(payload, "data"),
            model=_dict_value(payload, "model"),
            evaluation=_dict_value(payload, "evaluation"),
            validation=_dict_value(payload, "validation"),
            timing=_dict_value(payload, "timing"),
            artifacts=tuple(_dict_item(item, "artifacts") for item in artifacts),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run": dict(self.run),
            "data": dict(self.data),
            "model": dict(self.model),
            "evaluation": dict(self.evaluation),
            "validation": dict(self.validation),
            "timing": dict(self.timing),
            "artifacts": [dict(item) for item in self.artifacts],
        }


def _dict_value(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    return _dict_item(payload.get(key), key)


def _dict_item(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"trial run manifest {name} must be a mapping")
    return dict(value)


__all__ = ["TrialRunManifest"]
