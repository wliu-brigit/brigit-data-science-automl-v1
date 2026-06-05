"""Agent proposer context assembly."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from automl.data.profile import get_profile
from automl.data.selection import resolve_active_dataset
from automl.experiment.views import leaderboard, recent_failures, strategies_attempted
from automl.mlflow import client as mlflow_client
from automl.mlflow import experiment as mlflow_experiment
from automl.mlflow import project as mlflow_project
from automl.mlflow.trial.artifacts import load_error_report
from automl.model.preprocessing import describe_required_transformers
from automl.project import Session, allowed_dependencies, session as active_project_session
from automl.trial.types import TrialSummary


def gather_proposer_context(
    *,
    metric: str | None = None,
    n_top: int = 10,
    n_failures: int = 3,
    session: Session | None = None,
) -> dict[str, Any]:
    """Return the heterogeneous JSON packet consumed by the proposer agent."""

    active = session if session is not None else active_project_session()
    ranked = leaderboard(metric=metric, n=n_top, session=active)
    resolved_metric = ranked.metric
    human = leaderboard(
        metric=resolved_metric,
        n=n_top,
        training_origin="human",
        session=active,
    )
    failures = recent_failures(n=n_failures, session=active)
    strategy_counts = strategies_attempted(session=active)
    overview = _experiment_overview(active)
    dataset_context = _data_context(active, _trial_rows(ranked, human, failures))

    packet: dict[str, Any] = {
        "schema_version": 1,
        "project": active.project_name,
        "project_name": active.project_name,
        "experiment_id": active.active_experiment_id,
        "mlflow_experiment_id": _mlflow_experiment_id(active),
        "metric": resolved_metric,
        "higher_is_better": True,
        "dry_run": active.dry_run,
        "namespace": active.namespace,
        "project_instructions": _project_instructions(active),
        "overview": _as_dict(overview),
        "leaderboard": ranked.to_dict(),
        "human_trials": [row.to_dict() for row in human.rows],
        "recent_failures": [_failure_row(row) for row in failures],
        "strategies_attempted": strategy_counts,
        "trial_count": len(ranked.rows) + ranked.n_unscored,
        "prior_experiment": None,
        "project_contract": _project_contract(active),
        "data_context": dataset_context,
        "environment": {
            "allowed_dependencies": allowed_dependencies(session=active),
        },
    }
    if not ranked.rows:
        packet["prior_experiment"] = find_prior_experiment(session=active)
    return packet


def _project_contract(active: Session) -> dict[str, Any]:
    return {
        "target_column": active.config.target_column,
        "raw_target_column": active.config.raw_target_column,
        "primary_metric": active.config.primary_metric,
        "required_transformers": describe_required_transformers(session=active),
    }


def _failure_row(row: TrialSummary) -> dict[str, Any]:
    payload = row.to_dict()
    error = _failure_error_summary(row.run_id)
    if error:
        payload["error"] = error
    return payload


def _failure_error_summary(run_id: str) -> dict[str, Any]:
    if not run_id:
        return {}
    try:
        report = load_error_report(run_id)
    except Exception:
        return {}
    summary = {
        key: report[key]
        for key in (
            "runner_kind",
            "phase",
            "error_class",
            "message",
            "traceback_artifact",
            "proposal_artifact",
        )
        if key in report
    }
    traceback_tail = report.get("traceback_tail")
    if isinstance(traceback_tail, list):
        summary["traceback_tail"] = traceback_tail[-12:]
    return summary


def find_prior_experiment(*, session: Session | None = None) -> dict[str, Any] | None:
    """Return the newest prior experiment overview for the active project."""

    active = session if session is not None else active_project_session()
    candidates: list[dict[str, Any]] = []
    with mlflow_client.bound_for(active):
        for experiment_id in mlflow_project.list_experiments():
            if experiment_id == active.active_experiment_id:
                continue
            overview = mlflow_experiment.read_overview(experiment_id)
            if overview is None:
                continue
            candidates.append(
                {
                    "experiment_id": overview.experiment_id or experiment_id,
                    "project_name": overview.project_name or active.project_name,
                    "created_at": overview.created_at,
                }
            )
    candidates.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    return candidates[0] if candidates else None


def _experiment_overview(active: Session) -> Any | None:
    with mlflow_client.bound_for(active, experiment_id=active.active_experiment_id):
        return mlflow_experiment.read_overview(active.active_experiment_id)


def _mlflow_experiment_id(active: Session) -> str | None:
    with mlflow_client.bound_for(active, experiment_id=active.active_experiment_id):
        return mlflow_experiment.mlflow_experiment_id(active.active_experiment_id)


def _data_context(active: Session, trial_rows: list[TrialSummary]) -> dict[str, Any]:
    active_dataset = resolve_active_dataset(session=active)
    profile = get_profile(dataset_id=active_dataset.id, session=active)
    usage = _dataset_usage(trial_rows)
    active_usage = usage.get(active_dataset.identity_hash, 0)
    return {
        "active_dataset": active_dataset.to_dict(),
        "profile": profile.to_dict() if profile is not None else None,
        "dataset_usage": usage,
        "trial_usage_count": active_usage,
    }


def _dataset_usage(trial_rows: list[TrialSummary]) -> dict[str, int]:
    usage: dict[str, int] = {}
    for row in trial_rows:
        if row.dataset_hash:
            usage[row.dataset_hash] = usage.get(row.dataset_hash, 0) + 1
    return usage


def _trial_rows(*collections: Any) -> list[TrialSummary]:
    rows: list[TrialSummary] = []
    seen: set[str] = set()
    for collection in collections:
        current = getattr(collection, "rows", collection)
        for row in current:
            if not isinstance(row, TrialSummary):
                continue
            key = row.run_id or f"{row.slug}:{row.started_at}"
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def _project_instructions(active: Session) -> str:
    path = active.config.instructions_path
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _as_dict(value: Any) -> Any:
    if value is None:
        return None
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if is_dataclass(value):
        return asdict(value)
    return value


__all__ = ["find_prior_experiment", "gather_proposer_context"]
