"""Experiment leaderboard view."""

from __future__ import annotations

from automl.mlflow import client as mlflow_client
from automl.mlflow import experiment as mlflow_experiment
from automl.project import Session, session as active_project_session
from automl.experiment.views.types import LeaderboardData


def leaderboard(
    *,
    metric: str | None = None,
    n: int = 10,
    training_origin: str | None = None,
    session: Session | None = None,
) -> LeaderboardData:
    active = session if session is not None else active_project_session()
    resolved_metric = metric if metric is not None else _default_leaderboard_metric(active)
    with mlflow_client.bound_for(active, experiment_id=active.active_experiment_id):
        all_rows = mlflow_experiment.list_trials(training_origin=training_origin)
        scored_rows = tuple(
            mlflow_experiment.top_n_by_metric(
                resolved_metric,
                n=len(all_rows),
                training_origin=training_origin,
            )
        )
    return LeaderboardData(
        metric=resolved_metric,
        experiment_id=active.active_experiment_id,
        rows=scored_rows[:n],
        n_unscored=max(len(all_rows) - len(scored_rows), 0),
    )


__all__ = ["leaderboard"]


def _default_leaderboard_metric(active: Session) -> str:
    eval_split = active.config.run_config.eval_split if active.config.run_config is not None else "test"
    primary_metric = active.config.primary_metric if active.config.eval_spec is not None else "auc"
    return f"eval.{eval_split}.{primary_metric}"
