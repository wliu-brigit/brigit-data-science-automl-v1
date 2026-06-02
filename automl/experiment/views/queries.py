"""Experiment query helpers."""

from __future__ import annotations

from automl.mlflow import client as mlflow_client
from automl.mlflow import experiment as mlflow_experiment
from automl.project import Session, session as active_project_session
from automl.trial.types import TrialStatus, TrialSummary


def recent_failures(
    n: int = 3,
    *,
    training_origin: str | None = None,
    session: Session | None = None,
) -> list[TrialSummary]:
    active = session if session is not None else active_project_session()
    with mlflow_client.bound_for(active, experiment_id=active.active_experiment_id):
        return list(
            mlflow_experiment.list_trials(
                status=TrialStatus.FAILED,
                limit=n,
                training_origin=training_origin,
            )
        )


def strategies_attempted(*, session: Session | None = None) -> dict[str, int]:
    active = session if session is not None else active_project_session()
    with mlflow_client.bound_for(active, experiment_id=active.active_experiment_id):
        rows = mlflow_experiment.list_trials()
    counts: dict[str, int] = {}
    for row in rows:
        if row.strategy:
            counts[row.strategy] = counts.get(row.strategy, 0) + 1
    return counts


__all__ = ["recent_failures", "strategies_attempted"]
