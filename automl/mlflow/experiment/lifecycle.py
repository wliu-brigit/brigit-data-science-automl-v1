"""Experiment-level lifecycle operations."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from automl.errors import StorageError
from automl.mlflow import _routing
from automl.mlflow import client
from automl.mlflow import project as mlflow_project
from automl.mlflow import tags
from automl.project.overview import ProjectOverview

if TYPE_CHECKING:
    from automl.experiment.store import ExperimentOverview


def ensure(experiment_id: str | None = None) -> None:
    """Create the routed MLflow experiment if absent."""
    resolved_experiment_id = _resolved_experiment_id(experiment_id)
    try:
        mlflow_client = client.raw()
        name = _routing.experiment_route(experiment_id)
        existing = mlflow_client.get_experiment_by_name(name)
        if existing is not None:
            if getattr(existing, "lifecycle_stage", "active") == "deleted":
                raise StorageError(
                    f"MLflow experiment {name!r} is deleted; purge it or choose another id"
                )
            _ensure_project_overview(resolved_experiment_id)
            return
        mlflow_experiment_id = mlflow_client.create_experiment(name)
        mlflow_client.set_experiment_tag(
            mlflow_experiment_id,
            tags.CREATED_BY,
            "brigit-automl",
        )
        _ensure_project_overview(resolved_experiment_id)
    except StorageError:
        raise
    except Exception as exc:
        raise StorageError("Failed to ensure MLflow experiment") from exc


def mlflow_experiment_id(experiment_id: str | None = None) -> str | None:
    """Return the numeric MLflow experiment id for the routed experiment, if present."""
    try:
        found = client.raw().get_experiment_by_name(_routing.experiment_route(experiment_id))
    except Exception as exc:
        raise StorageError("Failed to resolve MLflow experiment") from exc
    if found is None:
        return None
    return str(found.experiment_id)


def ensure_overview(experiment_id: str | None = None) -> ExperimentOverview:
    ensure(experiment_id)
    overview = read_overview(experiment_id)
    if overview is not None:
        return overview
    numeric_experiment_id = mlflow_experiment_id(experiment_id)
    if numeric_experiment_id is None:
        raise StorageError("Failed to resolve MLflow experiment for overview")
    bound = client.bound()
    resolved_experiment_id = experiment_id or bound.experiment_id or ""
    created_at = datetime.now(UTC).isoformat()
    try:
        run = client.raw().create_run(
            numeric_experiment_id,
            tags=client.context_tags(
                {
                    tags.RUN_KIND: "experiment_overview",
                    tags.EXPERIMENT_ID: str(resolved_experiment_id),
                    tags.PROJECT_NAME: bound.project_name,
                    "run.dry_run": str(bound.dry_run),
                    tags.CREATED_AT: created_at,
                    "mlflow.runName": "000_overview",
                }
            ),
        )
        client.raw().set_terminated(run.info.run_id, status="FINISHED")
        client.remember_run_experiment(run.info.run_id, str(numeric_experiment_id))
        return _experiment_overview_from_run(run)
    except Exception as exc:
        raise StorageError("Failed to create MLflow experiment overview") from exc


def read_overview(experiment_id: str | None = None) -> ExperimentOverview | None:
    numeric_experiment_id = mlflow_experiment_id(experiment_id)
    if numeric_experiment_id is None:
        return None
    try:
        runs = client.raw().search_runs(
            [numeric_experiment_id],
            filter_string=f"tags.{tags.RUN_KIND} = 'experiment_overview'",
            max_results=1,
        )
    except Exception as exc:
        raise StorageError("Failed to read MLflow experiment overview") from exc
    if not runs:
        return None
    return _experiment_overview_from_run(list(runs)[0])


def write_overview(overview: object, experiment_id: str | None = None) -> None:
    ensure(experiment_id)


def get_active_dataset(experiment_id: str | None = None) -> str | None:
    numeric_experiment_id = mlflow_experiment_id(experiment_id)
    if numeric_experiment_id is None:
        return None
    try:
        experiment = client.raw().get_experiment(numeric_experiment_id)
    except Exception as exc:
        raise StorageError("Failed to read active dataset") from exc
    if experiment is None:
        return None
    return experiment.tags.get(tags.ACTIVE_DATASET_ID)


def set_active_dataset(dataset_id: str, experiment_id: str | None = None) -> None:
    ensure(experiment_id)
    numeric_experiment_id = mlflow_experiment_id(experiment_id)
    if numeric_experiment_id is None:
        raise StorageError("Failed to resolve MLflow experiment for active dataset")
    try:
        client.raw().set_experiment_tag(
            numeric_experiment_id,
            tags.ACTIVE_DATASET_ID,
            dataset_id,
        )
    except Exception as exc:
        raise StorageError("Failed to set active dataset") from exc


def _resolved_experiment_id(experiment_id: str | None = None) -> str:
    return str(experiment_id or client.bound().experiment_id or "")


def _ensure_project_overview(experiment_id: str) -> None:
    if not experiment_id or experiment_id == "000_overview":
        return
    overview = mlflow_project.ensure_overview()
    mlflow_project.write_overview(
        ProjectOverview(
            project_name=overview.project_name,
            created_at=overview.created_at,
            current_experiment_id=experiment_id,
        )
    )


def _experiment_overview_from_run(run) -> ExperimentOverview:
    from automl.experiment.store import ExperimentOverview

    run_tags = run.data.tags
    return ExperimentOverview(
        run_id=str(run.info.run_id),
        experiment_id=str(run_tags.get(tags.EXPERIMENT_ID, "")),
        project_name=str(run_tags.get(tags.PROJECT_NAME, client.bound().project_name)),
        created_at=str(run_tags.get(tags.CREATED_AT, "")),
        dry_run=str(run_tags.get("run.dry_run", "False")).lower() == "true",
    )


__all__ = [
    "ensure",
    "ensure_overview",
    "get_active_dataset",
    "mlflow_experiment_id",
    "read_overview",
    "set_active_dataset",
    "write_overview",
]
