"""Experiment-scoped MLflow queries."""

from __future__ import annotations

from datetime import UTC, datetime

from automl.errors import StorageError
from automl.mlflow import client
from automl.mlflow import tags
from automl.mlflow.experiment.lifecycle import mlflow_experiment_id
from automl.trial.types import TrialStatus, TrialSummary


TRAINING_ORIGIN_TAG = tags.TRIAL_TRAINING_ORIGIN
HYPOTHESIS_TAG = tags.TRIAL_HYPOTHESIS
DATA_IDENTITY_HASH_TAG = "data.identity_hash"
TRAINING_TIME_METRIC = "timing.total_seconds"
N_FEATURES_METRIC = "model.n_features"


def next_trial_number(*, experiment_id: str | None = None) -> int:
    """Return max canonical trial-number tag in the routed experiment plus one."""
    numeric_experiment_id = mlflow_experiment_id(experiment_id)
    if numeric_experiment_id is None:
        return 1
    max_number = 0
    for run in _search_all_runs(
        [numeric_experiment_id],
        f"tags.{tags.RUN_KIND} = 'trial'",
    ):
        raw = str(run.data.tags.get(tags.TRIAL_NUMBER) or "")
        if raw.isdigit() and int(raw) >= 1:
            max_number = max(max_number, int(raw))
    return max_number + 1


def find_trial_run_id(trial_id: str, *, experiment_id: str | None = None) -> str:
    summaries = search_trials(
        f"tags.{tags.TRIAL_ID} = '{trial_id}'",
        experiment_id=experiment_id,
    )
    if not summaries:
        raise KeyError(f"trial {trial_id!r} not found")
    summaries.sort(key=lambda row: row.started_at or "", reverse=True)
    return summaries[0].run_id


def search_trials(
    filter_string: str,
    *,
    experiment_id: str | None = None,
    max_results: int = 1000,
) -> list[TrialSummary]:
    numeric_experiment_id = mlflow_experiment_id(experiment_id)
    if numeric_experiment_id is None:
        return []
    runs = _search_all_runs([numeric_experiment_id], filter_string, max_results=max_results)
    runs.sort(key=lambda run: getattr(run.info, "start_time", 0) or 0, reverse=True)
    return [_run_to_trial_summary(run) for run in runs]


def list_trials(
    experiment_id: str | None = None,
    *,
    limit: int | None = None,
    status: str | None = None,
    training_origin: str | None = None,
) -> list[TrialSummary]:
    filter_parts = [f"tags.{tags.RUN_KIND} = 'trial'"]
    if status:
        filter_parts.append(f"tags.{tags.TRIAL_STATUS} = '{_status_value(status)}'")
    runs = _trial_runs(
        " and ".join(filter_parts),
        experiment_id=experiment_id,
        training_origin=training_origin,
    )
    runs.sort(key=lambda run: getattr(run.info, "start_time", 0) or 0, reverse=True)
    rows = [_run_to_trial_summary(run) for run in runs]
    return rows[:limit] if limit is not None else rows


def top_n_by_metric(
    metric: str,
    n: int = 10,
    *,
    ascending: bool = False,
    experiment_id: str | None = None,
    training_origin: str | None = None,
) -> list[TrialSummary]:
    runs = [
        run
        for run in _trial_runs(experiment_id=experiment_id, training_origin=training_origin)
        if metric in run.data.metrics
    ]
    runs.sort(key=lambda run: run.data.metrics[metric], reverse=not ascending)
    return [
        _run_to_trial_summary(
            run,
            primary_metric_name=metric,
            primary_metric_value=run.data.metrics[metric],
        )
        for run in runs[:n]
    ]


def _trial_runs(
    filter_string: str | None = None,
    *,
    experiment_id: str | None = None,
    training_origin: str | None = None,
) -> list[object]:
    training_origin = _normalize_training_origin(training_origin)
    numeric_experiment_id = mlflow_experiment_id(experiment_id)
    if numeric_experiment_id is None:
        return []
    runs = _search_all_runs(
        [numeric_experiment_id],
        filter_string or f"tags.{tags.RUN_KIND} = 'trial'",
    )
    if training_origin is not None:
        runs = [
            run
            for run in runs
            if str(run.data.tags.get(TRAINING_ORIGIN_TAG, "")) == training_origin
        ]
    return runs


def _normalize_training_origin(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"", "all"}:
        return None
    return normalized


def _run_to_trial_summary(
    run: object,
    *,
    primary_metric_name: str = "",
    primary_metric_value: float | None = None,
) -> TrialSummary:
    run_tags = dict(getattr(run.data, "tags", {}))
    metrics = dict(getattr(run.data, "metrics", {}))
    return TrialSummary(
        run_id=str(getattr(run.info, "run_id", "")),
        slug=str(run_tags.get(tags.TRIAL_SLUG, "")),
        strategy=str(getattr(run.data, "params", {}).get(tags.TRIAL_STRATEGY, "")),
        status=_status_from_run(run),
        primary_metric_name=primary_metric_name,
        primary_metric_value=primary_metric_value,
        started_at=_ms_to_iso(getattr(run.info, "start_time", None)),
        ended_at=_ms_to_iso(getattr(run.info, "end_time", None)),
        parent_run_id=None,
        dataset_hash=_optional_str(run_tags.get(DATA_IDENTITY_HASH_TAG)),
        trial_number=_optional_int(run_tags.get(tags.TRIAL_NUMBER)),
        hypothesis=str(getattr(run.data, "params", {}).get(HYPOTHESIS_TAG, "")),
        training_origin=str(run_tags.get(TRAINING_ORIGIN_TAG, "")),
        training_time_s=_optional_float(metrics.get(TRAINING_TIME_METRIC)),
        n_features=_optional_int(metrics.get(N_FEATURES_METRIC)),
    )


def _status_from_run(run: object) -> TrialStatus:
    tagged = getattr(run.data, "tags", {}).get(tags.TRIAL_STATUS)
    raw = tagged or getattr(run.info, "status", "")
    try:
        return TrialStatus(str(raw).upper())
    except ValueError:
        return TrialStatus.UNKNOWN


def _status_value(value: object) -> str:
    if isinstance(value, TrialStatus):
        return value.value
    return str(value)


def _ms_to_iso(value: object) -> str | None:
    if value in (None, ""):
        return None
    return datetime.fromtimestamp(int(value) / 1000, UTC).isoformat().replace("+00:00", "Z")


def _optional_str(value: object) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    return int(value)  # type: ignore[arg-type]


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)  # type: ignore[arg-type]


def _search_all_runs(
    experiment_ids: list[str],
    filter_string: str,
    *,
    max_results: int = 1000,
) -> list[object]:
    try:
        mlflow_client = client.raw()
        runs: list[object] = []
        page_token = None
        while True:
            page = mlflow_client.search_runs(
                experiment_ids,
                filter_string=filter_string,
                max_results=max_results,
                page_token=page_token,
            ) or []
            runs.extend(list(page))
            page_token = getattr(page, "token", None) or getattr(page, "next_page_token", None)
            if not page_token:
                break
        return runs
    except Exception as exc:
        raise StorageError("Failed to search MLflow runs") from exc


__all__ = [
    "find_trial_run_id",
    "list_trials",
    "next_trial_number",
    "search_trials",
    "top_n_by_metric",
]
