"""Experiment summary view."""

from __future__ import annotations

from typing import Any

from automl.experiment.views.leaderboard import leaderboard
from automl.experiment.views.queries import recent_failures, strategies_attempted
from automl.mlflow import client as mlflow_client
from automl.mlflow import experiment as mlflow_experiment
from automl.mlflow import project as mlflow_project
from automl.project import Session, session as active_project_session


def load_mlflow_context(*, session: Session | None = None) -> dict[str, Any]:
    active = session if session is not None else active_project_session()
    with mlflow_client.bound_for(active, experiment_id=active.active_experiment_id):
        return {
            "experiment_id": active.active_experiment_id,
            "trials": mlflow_experiment.list_trials(),
            "leaderboard": leaderboard(session=active),
            "recent_failures": recent_failures(session=active),
            "strategies_attempted": strategies_attempted(session=active),
        }


def build_summary_from_context(context: dict[str, Any]) -> dict[str, Any]:
    trials = tuple(context.get("trials", ()))
    return {
        "summary_kind": "experiment_summary",
        "experiment_id": context.get("experiment_id", ""),
        "trial_count": len(trials),
        "finished_count": sum(1 for row in trials if row.status == "FINISHED"),
        "failed_count": sum(1 for row in trials if row.status == "FAILED"),
        "strategies_attempted": dict(context.get("strategies_attempted", {})),
        "recent_failures": [row.to_dict() for row in context.get("recent_failures", ())],
        "leaderboard": context.get("leaderboard").to_dict()
        if context.get("leaderboard") is not None
        else None,
    }


def build_summary(*, session: Session | None = None) -> dict[str, Any]:
    return build_summary_from_context(load_mlflow_context(session=session))


def experiments(*, session: Session | None = None) -> list[dict[str, Any]]:
    active = session if session is not None else active_project_session()
    with mlflow_client.bound_for(active):
        ids = mlflow_project.list_experiments()
        metric = active.config.primary_metric if active.config.eval_spec is not None else ""
        rows = []
        for experiment_id in ids:
            trials = mlflow_experiment.list_trials(experiment_id=experiment_id)
            top = (
                mlflow_experiment.top_n_by_metric(metric, n=1, experiment_id=experiment_id)
                if metric
                else []
            )
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "trial_count": len(trials),
                    "top_metric": metric,
                    "top_run_id": top[0].run_id if top else "",
                    "top_metric_value": top[0].primary_metric_value if top else None,
                }
            )
    return rows


__all__ = ["build_summary", "build_summary_from_context", "experiments", "load_mlflow_context"]
