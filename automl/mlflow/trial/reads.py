"""Trial read primitives for the MLflow seam."""

from __future__ import annotations

from automl.errors import StorageError
from automl.mlflow import client
from automl.mlflow import tags
from automl.trial.types import ArtifactRef, ParentExperimentRef, TrialDetails, TrialStatus


def get_details(run_id: str) -> TrialDetails:
    """Return cheap run details without loading evaluation artifact contents."""
    try:
        run = client.raw().get_run(run_id)
        tags = dict(run.data.tags)
        return TrialDetails(
            run_id=str(run.info.run_id),
            status=_status_from_run(run),
            params={str(key): str(value) for key, value in run.data.params.items()},
            metrics={str(key): float(value) for key, value in run.data.metrics.items()},
            tags={str(key): str(value) for key, value in tags.items()},
            artifacts=list_artifacts(run_id),
            evaluations=None,
        )
    except StorageError:
        raise
    except Exception as exc:
        raise StorageError(f"Failed to read MLflow trial details for run {run_id!r}") from exc


def get_metrics(run_id: str) -> dict[str, float]:
    try:
        return {
            str(key): float(value)
            for key, value in client.raw().get_run(run_id).data.metrics.items()
        }
    except Exception as exc:
        raise StorageError(f"Failed to read MLflow metrics for run {run_id!r}") from exc


def list_artifacts(run_id: str) -> tuple[ArtifactRef, ...]:
    try:
        return tuple(_walk_artifacts(run_id, ""))
    except Exception as exc:
        raise StorageError(f"Failed to list MLflow artifacts for run {run_id!r}") from exc


def get_parent_experiment(run_id: str) -> ParentExperimentRef:
    try:
        mlflow_client = client.raw()
        run = mlflow_client.get_run(run_id)
        experiment = mlflow_client.get_experiment(str(run.info.experiment_id))
        if experiment is None:
            raise StorageError(f"MLflow experiment {run.info.experiment_id!r} not found")
        parsed = _parse_experiment_route(str(experiment.name))
        return ParentExperimentRef(
            mlflow_experiment_id=str(experiment.experiment_id),
            mlflow_experiment_name=str(experiment.name),
            dry_run=parsed["dry_run"],
            project_name=parsed["project_name"],
            experiment_id=parsed["experiment_id"],
        )
    except StorageError:
        raise
    except Exception as exc:
        raise StorageError(f"Failed to read parent experiment for run {run_id!r}") from exc


def _walk_artifacts(run_id: str, path: str) -> list[ArtifactRef]:
    refs: list[ArtifactRef] = []
    for item in client.raw().list_artifacts(run_id, path or None):
        item_path = str(item.path)
        if getattr(item, "is_dir", False):
            refs.extend(_walk_artifacts(run_id, item_path))
        else:
            refs.append(
                ArtifactRef(
                    path=item_path,
                    file_size=getattr(item, "file_size", None),
                )
            )
    return refs


def _status_from_run(run: object) -> TrialStatus:
    tagged = getattr(run.data, "tags", {}).get(tags.TRIAL_STATUS)
    raw = tagged or getattr(run.info, "status", "")
    try:
        return TrialStatus(str(raw).upper())
    except ValueError:
        return TrialStatus.UNKNOWN


def _parse_experiment_route(name: str) -> dict[str, object]:
    bound = client.bound()
    remaining = [segment for segment in name.split("/") if segment]
    namespace_segments = [segment for segment in bound.namespace.strip("/").split("/") if segment]
    if namespace_segments and remaining[: len(namespace_segments)] == namespace_segments:
        remaining = remaining[len(namespace_segments) :]
    dry_run = bool(remaining and remaining[0] == "dry_run")
    if dry_run:
        remaining = remaining[1:]
    project_name = remaining[0] if len(remaining) >= 1 else ""
    experiment_id = remaining[1] if len(remaining) >= 2 else ""
    return {
        "dry_run": dry_run,
        "project_name": project_name,
        "experiment_id": experiment_id,
    }


__all__ = ["get_details", "get_metrics", "get_parent_experiment", "list_artifacts"]
