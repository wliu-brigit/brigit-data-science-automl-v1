"""Active dataset selection helpers."""

from __future__ import annotations

from automl.data.dataset import Dataset
from automl.errors import DataError
from automl.mlflow import client as mlflow_client
from automl.mlflow import experiment as mlflow_experiment
from automl.mlflow.experiment import artifacts as experiment_artifacts
from automl.project import Session
from automl.project import session as active_project_session


def activate_dataset(dataset_id: str, *, session: Session | None = None) -> Dataset:
    """Set the experiment active dataset after validating the record exists."""
    active = _session(session)
    with mlflow_client.bound_for(active, experiment_id=active.active_experiment_id):
        record = experiment_artifacts.read_dataset_record(
            dataset_id,
            experiment_id=active.active_experiment_id,
        )
        if record is None:
            raise KeyError(f"dataset {dataset_id!r} not found")
        dataset = Dataset.from_dict(record)
        mlflow_experiment.set_active_dataset(
            dataset.id,
            experiment_id=active.active_experiment_id,
        )
        experiment_artifacts.write_active_dataset_pointer(
            dataset.id,
            experiment_id=active.active_experiment_id,
        )
        return dataset


def resolve_active_dataset_id(*, session: Session | None = None) -> str:
    """Return the validated active dataset id for the current experiment."""
    active = _session(session)
    with mlflow_client.bound_for(active, experiment_id=active.active_experiment_id):
        return _resolve_active_dataset_id_bound(active)


def resolve_active_dataset(*, session: Session | None = None) -> Dataset:
    """Return the validated active dataset record for the current experiment."""
    active = _session(session)
    with mlflow_client.bound_for(active, experiment_id=active.active_experiment_id):
        dataset_id = _resolve_active_dataset_id_bound(active)
        record = experiment_artifacts.read_dataset_record(
            dataset_id,
            experiment_id=active.active_experiment_id,
        )
        if record is None:
            raise DataError(f"active dataset pointer points at missing dataset {dataset_id!r}")
        return Dataset.from_dict(record)


def _resolve_active_dataset_id_bound(active: Session) -> str:
    active_id = mlflow_experiment.get_active_dataset(
        experiment_id=active.active_experiment_id,
    )
    if not active_id:
        raise DataError("active dataset pointer is not set")
    pointer = experiment_artifacts.read_active_dataset_pointer(
        experiment_id=active.active_experiment_id,
    )
    if pointer is None:
        raise DataError("active dataset pointer artifact is missing")
    pointer_id = str(pointer.get("active_dataset_id") or "")
    if not pointer_id:
        raise DataError("active dataset pointer artifact is invalid")
    if pointer_id != active_id:
        raise DataError(
            f"active dataset pointer mismatch: tag={active_id!r}, artifact={pointer_id!r}"
        )
    record = experiment_artifacts.read_dataset_record(
        active_id,
        experiment_id=active.active_experiment_id,
    )
    if record is None:
        raise DataError(f"active dataset pointer points at missing dataset {active_id!r}")
    return active_id


def _session(explicit: Session | None) -> Session:
    return explicit if explicit is not None else active_project_session()


__all__ = ["activate_dataset", "resolve_active_dataset", "resolve_active_dataset_id"]
