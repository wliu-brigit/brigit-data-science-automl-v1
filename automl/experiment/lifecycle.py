"""Experiment lifecycle domain API."""

from __future__ import annotations

from automl.mlflow import client as mlflow_client
from automl.mlflow import experiment as mlflow_experiment
from automl.project import Session, session as active_project_session
from automl.experiment.store import ExperimentOverview


def create(
    experiment_id: str | None = None,
    *,
    session: Session | None = None,
) -> ExperimentOverview:
    """Create or read a logical experiment overview."""
    active = session if session is not None else _optional_active_session()
    with mlflow_client.bound_for(active, experiment_id=_target_experiment_id(active, experiment_id)):
        mlflow_experiment.ensure(experiment_id)
        return mlflow_experiment.ensure_overview(experiment_id)


def _optional_active_session() -> Session | None:
    try:
        return active_project_session()
    except Exception:
        return None


def _target_experiment_id(active: Session | None, experiment_id: str | None) -> str | None:
    if experiment_id is not None or active is None:
        return experiment_id
    return active.active_experiment_id


__all__ = ["create"]
