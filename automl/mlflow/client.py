"""Process-bound MLflow connection state and UI URL helpers."""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Iterator

from automl.errors import StorageError

if TYPE_CHECKING:
    from automl.project import Session


# MLflow's HTTP client retries 408/429/5xx with exponential backoff; its default
# budget (7 retries, factor 2) sleeps ~254s before surfacing an error. Tracking
# servers below MLflow 3.12 return a retryable 500 — not 404 — for a *missing*
# artifact (fixed upstream in 3.12.0, mlflow/mlflow#22310), so downloading an
# absent artifact burned that whole budget while appearing to hang. One retry
# keeps tolerance for a single transient blip and bounds the doomed case to a
# few seconds (measured ~5s against the production server). Module scope so
# every process that reaches MLflow through this seam (CLI verbs, agent hooks,
# the runner) shares one budget; ``setdefault`` respects an operator override.
HTTP_MAX_RETRIES = "1"
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", HTTP_MAX_RETRIES)


@dataclass(frozen=True)
class Bound:
    tracking_uri: str
    bucket: str
    gcs_prefix: str
    project_name: str
    experiment_id: str | None = None
    dry_run: bool = False
    namespace: str = ""


_BOUND: ContextVar[Bound | None] = ContextVar("automl_mlflow_bound", default=None)
_RUN_EXPERIMENT_IDS: dict[str, str] = {}


def bind(
    *,
    tracking_uri: str,
    bucket: str,
    gcs_prefix: str,
    project_name: str,
    experiment_id: str | None = None,
    dry_run: bool = False,
    namespace: str = "",
) -> None:
    """Set process-level MLflow and artifact-routing state."""
    _BOUND.set(
        Bound(
            tracking_uri=tracking_uri,
            bucket=bucket,
            gcs_prefix=gcs_prefix,
            project_name=project_name,
            experiment_id=experiment_id,
            dry_run=dry_run,
            namespace=namespace,
        )
    )


def clear() -> None:
    """Clear bound state for tests and subprocess setup."""
    _BOUND.set(None)
    _RUN_EXPERIMENT_IDS.clear()


def bound() -> Bound:
    """Return current MLflow binding or raise the seam's storage error."""
    current = _BOUND.get()
    if current is None:
        raise StorageError("MLflow not bound; call automl.use_project(...) first")
    return current


@contextmanager
def bound_for(active: Session | None, *, experiment_id: str | None = None) -> Iterator[None]:
    """Temporarily bind MLflow state for a project session."""
    if active is None:
        yield
        return
    try:
        prior = bound()
    except StorageError:
        prior = None
    bind(
        tracking_uri=active.config.mlflow_tracking_uri,
        bucket=active.config.gcs_bucket,
        gcs_prefix=active.config.gcs_prefix,
        project_name=active.project_name,
        experiment_id=experiment_id,
        dry_run=active.dry_run,
        namespace=active.namespace,
    )
    try:
        yield
    finally:
        if prior is None:
            clear()
        else:
            bind(
                tracking_uri=prior.tracking_uri,
                bucket=prior.bucket,
                gcs_prefix=prior.gcs_prefix,
                project_name=prior.project_name,
                experiment_id=prior.experiment_id,
                dry_run=prior.dry_run,
                namespace=prior.namespace,
            )


def raw():
    """Return the low-level PyPI MLflow client for the bound tracking URI."""
    import mlflow

    tracking_uri = bound().tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)


def remember_run_experiment(run_id: str, mlflow_experiment_id: str) -> None:
    """Record run ownership for URL helpers when later bound to an HTTP UI."""
    _RUN_EXPERIMENT_IDS[run_id] = mlflow_experiment_id


def run_url(run_id: str) -> str:
    """Return the MLflow UI URL for a run, or ``""`` for local tracking stores."""
    base = bound().tracking_uri.rstrip("/")
    if not base.startswith(("http://", "https://")):
        return ""
    experiment_id = _mlflow_experiment_id_for_run(run_id)
    return f"{base}/#/experiments/{experiment_id}/runs/{run_id}"


def artifact_url(run_id: str, artifact_path: str) -> str:
    """Return the MLflow UI URL for a run artifact, or ``""`` for local stores."""
    parent = run_url(run_id)
    if not parent:
        return ""
    return f"{parent}/artifacts/{artifact_path.strip('/')}"


def experiment_url(experiment_id: str | None = None) -> str:
    """Return the MLflow UI URL for the bound (or given) experiment.

    Returns ``""`` for local tracking stores, when no experiment is bound yet
    (exploration sessions), or when the experiment has not been created.
    """
    resolved = experiment_id if experiment_id is not None else bound().experiment_id
    if resolved is None:
        return ""
    from automl.mlflow import _routing

    return _experiment_route_url(_routing.experiment_route(resolved))


def project_url() -> str:
    """Return the MLflow UI URL for the project's overview experiment.

    The ``<project>/overview`` experiment is this system's project-level MLflow
    entity. Returns ``""`` for local stores or before that experiment exists.
    """
    from automl.mlflow import _routing

    return _experiment_route_url(_routing.experiment_route("overview"))


def _experiment_route_url(route: str) -> str:
    base = bound().tracking_uri.rstrip("/")
    if not base.startswith(("http://", "https://")):
        return ""
    experiment = get_experiment_by_name(route)
    if experiment is None:
        return ""
    return f"{base}/#/experiments/{experiment.experiment_id}"


def download_artifact(
    run_id: str,
    artifact_path: str,
    *,
    required: bool = False,
) -> str | None:
    """Download one run artifact; return its local path, or ``None`` when absent.

    Lists the artifact's parent directory first: ``list_artifacts`` reports a
    missing file cleanly and fast, while a blind ``download_artifacts`` of an
    absent path makes tracking servers below MLflow 3.12 return a retryable
    500 that burns the whole HTTP retry budget before failing
    (mlflow/mlflow#22310). With ``required=True`` absence raises the seam's
    storage error instead; transport failures always propagate for the
    caller's own wrapping.
    """
    mlflow_client = raw()
    parent = PurePosixPath(artifact_path).parent.as_posix()
    listed = mlflow_client.list_artifacts(run_id, None if parent == "." else parent)
    if artifact_path not in {item.path for item in listed}:
        if required:
            raise StorageError(
                f"MLflow artifact {artifact_path!r} not found on run {run_id!r}"
            )
        return None
    return mlflow_client.download_artifacts(run_id, artifact_path)


def log_artifact_file(run_id: str, artifact_path: str, local_path: Path) -> None:
    """Log a local file to a run artifact path through the MLflow seam."""
    parent = Path(artifact_path).parent.as_posix()
    try:
        raw().log_artifact(
            run_id,
            str(local_path),
            artifact_path=parent if parent != "." else None,
        )
    except Exception as exc:
        raise StorageError(f"Failed to log artifact {artifact_path!r}") from exc


def get_experiment_by_name(name: str):
    """Return an MLflow experiment by name through the MLflow seam."""
    try:
        return raw().get_experiment_by_name(name)
    except Exception as exc:
        raise StorageError(f"Failed to get MLflow experiment {name!r}") from exc


def delete_experiment(experiment_id: str) -> None:
    """Soft-delete an MLflow experiment through the MLflow seam."""
    try:
        raw().delete_experiment(experiment_id)
    except Exception as exc:
        raise StorageError(f"Failed to delete MLflow experiment {experiment_id!r}") from exc


def delete_run(run_id: str) -> None:
    """Soft-delete an MLflow run through the MLflow seam."""
    try:
        raw().delete_run(run_id)
    except Exception as exc:
        raise StorageError(f"Failed to delete MLflow run {run_id!r}") from exc


def run_start_time(run_id: str) -> int | None:
    """Return an MLflow run start timestamp in milliseconds, if present."""
    try:
        run = raw().get_run(run_id)
        return getattr(run.info, "start_time", None)
    except Exception as exc:
        raise StorageError(f"Failed to read MLflow run {run_id!r}") from exc


def _mlflow_experiment_id_for_run(run_id: str) -> str:
    cached = _RUN_EXPERIMENT_IDS.get(run_id)
    if cached:
        return cached
    try:
        run = raw().get_run(run_id)
    except Exception as exc:  # pragma: no cover - backend-specific transport shape
        raise StorageError(f"Could not resolve MLflow experiment id for run {run_id!r}") from exc
    experiment_id = str(run.info.experiment_id)
    remember_run_experiment(run_id, experiment_id)
    return experiment_id


__all__ = [
    "Bound",
    "HTTP_MAX_RETRIES",
    "artifact_url",
    "bind",
    "bound",
    "bound_for",
    "clear",
    "delete_experiment",
    "delete_run",
    "download_artifact",
    "experiment_url",
    "get_experiment_by_name",
    "log_artifact_file",
    "project_url",
    "raw",
    "remember_run_experiment",
    "run_start_time",
    "run_url",
]
