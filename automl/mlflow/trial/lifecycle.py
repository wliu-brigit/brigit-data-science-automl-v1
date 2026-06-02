"""Trial run lifecycle operations."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from automl.errors import StorageError
from automl.mlflow import client
from automl.mlflow import tags
from automl.mlflow.experiment.lifecycle import ensure_overview, mlflow_experiment_id


@contextmanager
def active(
    *,
    slug: str,
    strategy: str,
    parent_run_id: str | None = None,
    experiment_id: str | None = None,
) -> Iterator[str]:
    """Open a trial run and terminate it based on context outcome."""
    run_id = start(
        slug=slug,
        strategy=strategy,
        parent_run_id=parent_run_id,
        experiment_id=experiment_id,
    )
    try:
        yield run_id
    except Exception:
        end(run_id, "FAILED")
        raise
    else:
        end(run_id, "FINISHED")


def start(
    *,
    slug: str,
    strategy: str,
    parent_run_id: str | None = None,
    experiment_id: str | None = None,
) -> str:
    """Create a trial run without auto-ending it."""
    overview = ensure_overview(experiment_id)
    numeric_experiment_id = mlflow_experiment_id(experiment_id)
    if numeric_experiment_id is None:
        raise StorageError("MLflow experiment not found after ensure")
    bound = client.bound()
    resolved_experiment_id = experiment_id or bound.experiment_id
    try:
        mlflow_client = client.raw()
        run_tags = {
            tags.RUN_KIND: "trial",
            tags.PROJECT_NAME: bound.project_name,
            tags.EXPERIMENT_ID: str(resolved_experiment_id),
            tags.EXPERIMENT_OVERVIEW_RUN_ID: parent_run_id or overview.run_id,
            tags.TRIAL_SLUG: slug,
            tags.TRIAL_STATUS: "RUNNING",
            "mlflow.runName": slug,
        }
        run = mlflow_client.create_run(numeric_experiment_id, tags=run_tags)
        run_id = run.info.run_id
        mlflow_client.log_param(run_id, tags.TRIAL_STRATEGY, str(strategy))
        client.remember_run_experiment(run_id, str(numeric_experiment_id))
        return run_id
    except Exception as exc:
        raise StorageError("Failed to start MLflow trial run") from exc


def end(run_id: str, status: str) -> None:
    """Set trial status tag and terminate the MLflow run."""
    try:
        mlflow_client = client.raw()
        mlflow_client.set_tag(run_id, tags.TRIAL_STATUS, str(status))
        mlflow_client.set_terminated(run_id, status=str(status))
    except Exception as exc:
        raise StorageError("Failed to end MLflow trial run") from exc


__all__ = ["active", "end", "start"]
