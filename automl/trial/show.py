"""Trial read-domain helpers."""

from __future__ import annotations

from dataclasses import replace

from automl.mlflow import client as mlflow_client
from automl.mlflow import trial as mlflow_trial
from automl.project import Session
from automl.trial.types import TrialDetails


def show_trial(run_id: str, *, session: Session | None = None) -> TrialDetails:
    """Return a deep trial read with evaluation artifacts loaded."""
    with mlflow_client.bound_for(session, experiment_id=_active_experiment_id(session)):
        details = mlflow_trial.get_details(run_id)
        evaluations = tuple(
            mlflow_trial.artifacts.load_eval(run_id, label)
            for label, _eval_dataset_id in mlflow_trial.artifacts.list_eval(run_id)
        )
        return replace(details, evaluations=evaluations)


def load_model(run_id: str, *, session: Session | None = None):
    """Load the model artifact packaged for a trial run."""
    with mlflow_client.bound_for(session, experiment_id=_active_experiment_id(session)):
        return mlflow_trial.artifacts.load_model(run_id)


def _active_experiment_id(active: Session | None) -> str | None:
    return None if active is None else active.active_experiment_id


__all__ = ["load_model", "show_trial"]
