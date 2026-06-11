"""Typed run-cycle configuration for an AutoML project."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from automl.project.predicates import Predicate, Where


SAFE_ROUTE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
ALLOWED_EFFORTS = frozenset({"low", "medium", "high"})
DEFAULT_SPLIT_PREDICATES = {
    "train": Where("SPLIT_PCT") < 80,
    "test": Where("SPLIT_PCT") >= 80,
}


@dataclass(frozen=True, init=False)
class Splits:
    """Named, durable row-criteria over an immutable dataset.

    Values are Predicate expressions (see automl.project.predicates).
    Overlap is deliberately not policed — the harness records exactly what
    each named split meant for any trial and enforces nothing about
    disjointness (design §12).
    """

    predicates: Mapping[str, Predicate]

    def __init__(
        self,
        predicates: Mapping[str, Predicate] | None = None,
        *,
        train: Predicate | None = None,
        test: Predicate | None = None,
        **named: Predicate,
    ) -> None:
        raw: dict[str, Predicate] = {}
        has_explicit = (
            predicates is not None or train is not None or test is not None or bool(named)
        )
        if predicates is not None:
            raw.update(dict(predicates))
        if train is not None:
            raw["train"] = train
        if test is not None:
            raw["test"] = test
        raw.update(named)
        if not raw:
            if has_explicit:
                raise ValueError("Splits must define at least one named predicate")
            raw.update(DEFAULT_SPLIT_PREDICATES)
        for name, value in raw.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"split name must be a non-empty string, got {name!r}")
            if not isinstance(value, Predicate):
                raise TypeError(
                    f"split {name!r} must be a Where(...) predicate, got "
                    f"{type(value).__name__} — bucket ranges were removed; "
                    f'use Where("SPLIT_PCT") < 80'
                )
        object.__setattr__(self, "predicates", dict(raw))

    def resolve(self, name: str) -> Predicate:
        try:
            return self.predicates[name]
        except KeyError as exc:
            known = ", ".join(sorted(self.predicates))
            raise KeyError(f"split {name!r} is not defined; known splits: {known}") from exc

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "Splits":
        if not isinstance(payload, Mapping):
            raise TypeError(f"Splits payload must be a mapping, got {type(payload).__name__}")
        if "ranges" in payload:
            # Loud tombstone, not a fallback: pre-step-4 payloads are dead.
            raise ValueError(
                "Splits payload carries 'ranges' — bucket ranges were removed; "
                "re-serialize from Where(...) predicates"
            )
        raw = payload.get("predicates")
        if not isinstance(raw, Mapping):
            raise ValueError("Splits payload must contain a 'predicates' mapping")
        return cls({str(name): Predicate.from_dict(ast) for name, ast in raw.items()})

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicates": {
                name: predicate.to_dict() for name, predicate in self.predicates.items()
            }
        }


@dataclass(frozen=True)
class ModelRoute:
    """A single agent route."""

    model: str
    effort: str

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError(f"model must be a non-empty string, got {self.model!r}")
        if self.effort not in ALLOWED_EFFORTS:
            allowed = ", ".join(sorted(ALLOWED_EFFORTS))
            raise ValueError(f"effort must be one of {allowed}; got {self.effort!r}")


@dataclass(frozen=True)
class ModelsConfig:
    """Routes for the three agent roles."""

    manager: ModelRoute
    proposer: ModelRoute
    coder: ModelRoute

    def __post_init__(self) -> None:
        for role in ("manager", "proposer", "coder"):
            value = getattr(self, role)
            if not isinstance(value, ModelRoute):
                raise TypeError(f"models.{role} must be a ModelRoute, got {type(value).__name__}")


@dataclass(frozen=True, init=False)
class RunConfig:
    """Operational settings for an AutoML run cycle."""

    experiment_id: str
    splits: Splits
    models: ModelsConfig
    per_trial_seconds: int
    train_split: str
    eval_split: str
    serving_validation_seconds: int

    def __init__(
        self,
        *,
        experiment_id: str,
        splits: Splits | None = None,
        models: ModelsConfig,
        per_trial_seconds: int,
        train_split: str = "train",
        eval_split: str = "test",
        serving_validation_seconds: int = 300,
    ) -> None:
        chosen_splits = splits if splits is not None else Splits()
        if not isinstance(chosen_splits, Splits):
            raise TypeError(f"splits must be a Splits, got {type(chosen_splits).__name__}")
        if not isinstance(models, ModelsConfig):
            raise TypeError(f"models must be a ModelsConfig, got {type(models).__name__}")
        if not isinstance(experiment_id, str) or not experiment_id.strip():
            raise ValueError(f"experiment_id must be a non-empty string, got {experiment_id!r}")
        if not SAFE_ROUTE_COMPONENT_RE.fullmatch(experiment_id):
            raise ValueError(
                f"experiment_id must match {SAFE_ROUTE_COMPONENT_RE.pattern}, got {experiment_id!r}"
            )
        if (
            isinstance(per_trial_seconds, bool)
            or not isinstance(per_trial_seconds, int)
            or per_trial_seconds < 1
        ):
            raise ValueError(
                f"per_trial_seconds must be a positive integer, got {per_trial_seconds!r}"
            )
        if (
            isinstance(serving_validation_seconds, bool)
            or not isinstance(serving_validation_seconds, int)
            or serving_validation_seconds < 1
        ):
            raise ValueError(
                f"serving_validation_seconds must be a positive integer, got {serving_validation_seconds!r}"
            )
        chosen_splits.resolve(train_split)
        chosen_splits.resolve(eval_split)

        object.__setattr__(self, "experiment_id", experiment_id)
        object.__setattr__(self, "splits", chosen_splits)
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "per_trial_seconds", per_trial_seconds)
        object.__setattr__(self, "train_split", train_split)
        object.__setattr__(self, "eval_split", eval_split)
        object.__setattr__(self, "serving_validation_seconds", serving_validation_seconds)

__all__ = ["ModelRoute", "ModelsConfig", "RunConfig", "Splits"]
