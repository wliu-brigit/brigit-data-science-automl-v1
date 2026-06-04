"""Metric and evaluation specification contracts."""

from __future__ import annotations

import copy
import math
import numbers
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def _snake_case(name: str) -> str:
    first = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", first).lower()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(inner) for key, inner in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(inner) for inner in value]
    if isinstance(value, list):
        return [_jsonable(inner) for inner in value]
    return value


def is_scalar_value(value: Any) -> bool:
    if isinstance(value, np.generic):
        value = value.item()
    return isinstance(value, numbers.Real) and math.isfinite(float(value))


class Metric:
    name: str | None = None
    required_columns: tuple[str, ...] = ()
    required_augmentations: tuple[str, ...] = ()
    _sign: int = 1
    _alias: str | None = None

    @property
    def metric_name(self) -> str:
        return self.resolved_name()

    def __neg__(self):
        copied = copy.copy(self)
        copied._sign = -1 * self._sign
        return copied

    def with_alias(self, alias: str):
        copied = copy.copy(self)
        copied._alias = alias
        return copied

    def resolved_name(self) -> str:
        if self._alias:
            return self._alias
        base_name = self.name or _snake_case(type(self).__name__)
        if self._sign < 0:
            return f"negative_{base_name}"
        return base_name

    def compute(self, df: pd.DataFrame, y_pred: Any, target_col: str) -> Any:
        raise NotImplementedError

    def evaluate(self, df: pd.DataFrame, y_pred: Any, target_col: str) -> dict[str, Any]:
        value = self.compute(df, y_pred, target_col)
        if self._sign < 0:
            if not is_scalar_value(value):
                raise TypeError(
                    f"Metric '{self.resolved_name()}' cannot be negated because it is not scalar"
                )
            value = -float(value)
        return {"name": self.resolved_name(), "value": _jsonable(value)}


MetricLike = Metric | Mapping[str, Metric]


def _resolve_metric(metric: MetricLike) -> Metric:
    if isinstance(metric, Metric):
        return metric
    if not isinstance(metric, Mapping) or len(metric) != 1:
        raise ValueError("Metric aliases must be a mapping with exactly one item")
    alias, inner = next(iter(metric.items()))
    if not isinstance(alias, str) or not alias:
        raise ValueError("Metric alias must be a non-empty string")
    if not isinstance(inner, Metric):
        raise TypeError("Metric alias value must be a Metric")
    return inner.with_alias(alias)


def _required_attr(metric: Metric, attr: str) -> tuple[str, ...]:
    value = getattr(metric, attr, ())
    if callable(value):
        value = value()
    return tuple(str(item) for item in value)


def _unique_key_columns(unique_key: str | Sequence[str] | None) -> tuple[str, ...]:
    if unique_key is None:
        raise ValueError("unique_key is required when joining augmentation frames")
    if isinstance(unique_key, str):
        return (unique_key,)
    return tuple(str(column) for column in unique_key)


class EvalSpec:
    def __init__(self, *, primary: MetricLike, metrics: Sequence[MetricLike] = ()) -> None:
        self.primary = _resolve_metric(primary)
        self._extra_metrics = tuple(_resolve_metric(metric) for metric in metrics)
        names = [metric.resolved_name() for metric in self.metrics]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate metric name(s) in EvalSpec: {duplicates}")

    @property
    def primary_name(self) -> str:
        return self.primary.metric_name

    @property
    def metrics(self) -> tuple[Metric, ...]:
        return (self.primary, *self._extra_metrics)

    def required_columns(self) -> tuple[str, ...]:
        columns: list[str] = []
        for metric in self.metrics:
            for column in _required_attr(metric, "required_columns"):
                if column not in columns:
                    columns.append(column)
        return tuple(columns)

    def required_augmentations(self) -> tuple[str, ...]:
        names: list[str] = []
        for metric in self.metrics:
            for name in _required_attr(metric, "required_augmentations"):
                if name not in names:
                    names.append(name)
        return tuple(names)

    def validate_columns(self, df: pd.DataFrame, target_col: str) -> None:
        required = (target_col, *self.required_columns())
        missing = sorted(column for column in required if column not in df.columns)
        if missing:
            raise ValueError(f"missing required eval column(s): {missing}")

    def evaluate(
        self,
        df: pd.DataFrame,
        y_pred: Any,
        target_col: str,
        *,
        augmentation_frames: Mapping[str, pd.DataFrame] | None = None,
        unique_key: str | Sequence[str] | None = None,
    ) -> dict[str, object]:
        df_for_metrics = self._with_augmentation_frames(
            df,
            augmentation_frames=augmentation_frames,
            unique_key=unique_key,
        )
        self.validate_columns(df_for_metrics, target_col)
        records = []
        for metric in self.metrics:
            record = metric.evaluate(df_for_metrics, y_pred, target_col)
            record["augmentations"] = list(_required_attr(metric, "required_augmentations"))
            records.append(record)
        primary_value = records[0]["value"]
        if not is_scalar_value(primary_value):
            raise ValueError("primary metric must be a finite scalar")
        records[0]["value"] = float(primary_value)
        return {"primary": records[0]["name"], "metrics": records}

    def _with_augmentation_frames(
        self,
        df: pd.DataFrame,
        *,
        augmentation_frames: Mapping[str, pd.DataFrame] | None,
        unique_key: str | Sequence[str] | None,
    ) -> pd.DataFrame:
        needed = self.required_augmentations()
        if not needed:
            return df
        frames = augmentation_frames or {}
        missing = [name for name in needed if name not in frames]
        if missing:
            raise ValueError(f"required augmentations missing: {missing}")
        keys = _unique_key_columns(unique_key)
        joined = df
        for name in needed:
            frame = frames[name]
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"augmentation {name!r} must be a pandas DataFrame")
            missing_keys = [key for key in keys if key not in frame.columns]
            if missing_keys:
                raise KeyError(f"augmentation {name!r} missing unique_key columns: {missing_keys}")
            duplicate_rows = frame.duplicated(subset=list(keys), keep=False)
            if bool(duplicate_rows.any()):
                raise ValueError(f"augmentation {name!r} contains duplicate unique_key rows")
            overlap = sorted(set(frame.columns) & set(joined.columns) - set(keys))
            if overlap:
                raise ValueError(
                    f"augmentation {name!r} columns overlap existing evaluation columns: {overlap}"
                )
            joined = joined.merge(frame, on=list(keys), how="left", validate="one_to_one")
        return joined


def scalar_metric_records(result: Mapping[str, Any]) -> dict[str, float]:
    scalars: dict[str, float] = {}
    for record in result.get("metrics", []):
        if not isinstance(record, Mapping):
            continue
        name = str(record.get("name") or "")
        value = record.get("value")
        if name and is_scalar_value(value):
            scalars[name] = float(value)
    return scalars


__all__ = ["EvalSpec", "Metric", "is_scalar_value", "scalar_metric_records"]
