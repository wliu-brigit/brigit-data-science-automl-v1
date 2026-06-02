"""Typed run-cycle configuration for an AutoML project."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


SAFE_ROUTE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
ALLOWED_EFFORTS = frozenset({"low", "medium", "high"})
DEFAULT_SPLIT_RANGES = {
    "train": ((0, 80),),
    "test": ((80, 100),),
}


def _coerce_ranges(
    value: Sequence[tuple[int, int]], field_name: str
) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    seen_buckets: set[int] = set()
    for index, item in enumerate(value):
        try:
            low, high = item
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field_name}[{index}] must be a (low, high) pair, got {item!r}"
            ) from exc
        if (
            isinstance(low, bool)
            or isinstance(high, bool)
            or not isinstance(low, int)
            or not isinstance(high, int)
        ):
            raise ValueError(f"{field_name}[{index}] bounds must be ints, got {item!r}")
        if not (0 <= low <= 100 and 0 <= high <= 100):
            raise ValueError(f"{field_name}[{index}] bounds must be in [0, 100], got {item!r}")
        if low >= high:
            raise ValueError(f"{field_name}[{index}] must be ascending, got {item!r}")
        buckets = set(range(low, high))
        overlap = seen_buckets & buckets
        if overlap:
            raise ValueError(
                f"{field_name}[{index}] overlaps another range in {field_name!r} "
                f"at bucket {min(overlap)}"
            )
        seen_buckets.update(buckets)
        ranges.append((low, high))
    if not ranges:
        raise ValueError(f"{field_name} must contain at least one range")
    return tuple(ranges)


def _bucket_set(ranges: Sequence[tuple[int, int]]) -> set[int]:
    buckets: set[int] = set()
    for low, high in ranges:
        buckets.update(range(low, high))
    return buckets


def _validate_no_cross_name_overlap(ranges: Mapping[str, Sequence[tuple[int, int]]]) -> None:
    bucket_owner: dict[int, str] = {}
    for name, split_ranges in ranges.items():
        for bucket in _bucket_set(split_ranges):
            other = bucket_owner.get(bucket)
            if other is not None:
                raise ValueError(
                    f"split {name!r} overlaps split {other!r} at bucket {bucket}"
                )
            bucket_owner[bucket] = name


@dataclass(frozen=True, init=False)
class Splits:
    """Free-form named deterministic 0..99 bucket ranges."""

    ranges: Mapping[str, tuple[tuple[int, int], ...]]

    def __init__(
        self,
        ranges: Mapping[str, Sequence[tuple[int, int]]] | None = None,
        *,
        train: Sequence[tuple[int, int]] | None = None,
        test: Sequence[tuple[int, int]] | None = None,
        **named_ranges: Sequence[tuple[int, int]],
    ) -> None:
        raw: dict[str, Sequence[tuple[int, int]]] = {}
        has_explicit_ranges = (
            ranges is not None
            or train is not None
            or test is not None
            or bool(named_ranges)
        )
        if ranges is not None:
            raw.update(dict(ranges))
        if train is not None:
            raw["train"] = train
        if test is not None:
            raw["test"] = test
        raw.update(named_ranges)
        if not raw:
            if has_explicit_ranges:
                raise ValueError("Splits must define at least one named range")
            raw.update(DEFAULT_SPLIT_RANGES)

        coerced: dict[str, tuple[tuple[int, int], ...]] = {}
        for name, value in raw.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"split name must be a non-empty string, got {name!r}")
            coerced[name] = _coerce_ranges(value, name)
        _validate_no_cross_name_overlap(coerced)
        object.__setattr__(self, "ranges", coerced)

    def resolve(self, name: str) -> tuple[tuple[int, int], ...]:
        try:
            return self.ranges[name]
        except KeyError as exc:
            known = ", ".join(sorted(self.ranges))
            raise KeyError(f"split {name!r} is not defined; known splits: {known}") from exc

    def buckets(self, name: str) -> frozenset[int]:
        return frozenset(_bucket_set(self.resolve(name)))

    def train_buckets(self) -> frozenset[int]:
        return self.buckets("train")

    def test_buckets(self) -> frozenset[int]:
        return self.buckets("test")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "Splits":
        if not isinstance(payload, Mapping):
            raise TypeError(f"Splits payload must be a mapping, got {type(payload).__name__}")
        raw = payload.get("ranges", payload)
        if not isinstance(raw, Mapping):
            raise ValueError("Splits payload must contain a 'ranges' mapping")
        return cls(raw)

    def to_dict(self) -> dict[str, dict[str, list[list[int]]]]:
        return {
            "ranges": {
                name: [[low, high] for low, high in ranges]
                for name, ranges in self.ranges.items()
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

    def __init__(
        self,
        *,
        experiment_id: str,
        splits: Splits | None = None,
        models: ModelsConfig,
        per_trial_seconds: int,
        train_split: str = "train",
        eval_split: str = "test",
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
        chosen_splits.resolve(train_split)
        chosen_splits.resolve(eval_split)

        object.__setattr__(self, "experiment_id", experiment_id)
        object.__setattr__(self, "splits", chosen_splits)
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "per_trial_seconds", per_trial_seconds)
        object.__setattr__(self, "train_split", train_split)
        object.__setattr__(self, "eval_split", eval_split)

__all__ = ["ModelRoute", "ModelsConfig", "RunConfig", "Splits"]
