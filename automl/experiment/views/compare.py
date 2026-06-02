"""Experiment trial comparison view."""

from __future__ import annotations

import numbers

from automl.experiment.views.types import ComparisonResult, MetricDelta
from automl.project import Session
from automl.trial import show_trial


def compare(run_ids: list[str] | tuple[str, ...], *, session: Session | None = None) -> ComparisonResult:
    if not run_ids:
        raise ValueError("at least one run_id is required")
    runs = tuple(show_trial(run_id, session=session) for run_id in run_ids)
    metric_deltas = _metric_deltas(runs[:2]) if len(runs) >= 2 else ()
    return ComparisonResult(
        run_ids=tuple(str(run_id) for run_id in run_ids),
        runs=runs,
        metric_deltas=metric_deltas,
    )


def _metric_deltas(runs: tuple) -> tuple[MetricDelta, ...]:
    left, right = runs
    keys = sorted(set(left.metrics) | set(right.metrics))
    deltas: list[MetricDelta] = []
    for key in keys:
        value_a = _number_or_none(left.metrics.get(key))
        value_b = _number_or_none(right.metrics.get(key))
        delta = value_b - value_a if value_a is not None and value_b is not None else None
        deltas.append(MetricDelta(metric=key, value_a=value_a, value_b=value_b, delta=delta))
    return tuple(deltas)


def _number_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    return float(value)


__all__ = ["compare"]
