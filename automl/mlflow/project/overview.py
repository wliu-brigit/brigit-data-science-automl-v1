"""Project overview types and MLflow helpers."""

from __future__ import annotations

from datetime import UTC, datetime

from automl.errors import StorageError
from automl.mlflow import _routing
from automl.mlflow import client
from automl.mlflow import tags
from automl.project.overview import ProjectOverview


def read_overview() -> ProjectOverview | None:
    run = _overview_run()
    if run is None:
        return None
    return _overview_from_run(run)


def ensure_overview() -> ProjectOverview:
    run = _overview_run()
    if run is not None:
        return _overview_from_run(run)
    experiment_id = _ensure_overview_experiment()
    created_at = datetime.now(UTC).isoformat()
    try:
        run = client.raw().create_run(
            experiment_id,
            tags={
                tags.RUN_KIND: "project_overview",
                tags.PROJECT_NAME: client.bound().project_name,
                tags.CREATED_AT: created_at,
                "mlflow.runName": "overview",
            },
        )
        client.raw().set_terminated(run.info.run_id, status="FINISHED")
        client.remember_run_experiment(run.info.run_id, str(experiment_id))
        return ProjectOverview(
            project_name=client.bound().project_name,
            created_at=created_at,
        )
    except Exception as exc:
        raise StorageError("Failed to create project overview run") from exc


def write_overview(overview: ProjectOverview) -> None:
    run_id = _ensure_overview_run_id()
    try:
        client.raw().set_tag(run_id, "project.current_experiment_id", overview.current_experiment_id or "")
        client.raw().set_tag(run_id, "data.dataset_count", str(overview.dataset_count))
    except Exception as exc:
        raise StorageError("Failed to write project overview") from exc


def list_experiments() -> list[str]:
    try:
        root = f"{_routing.project_route()}/"
        ids: list[str] = []
        for experiment in client.raw().search_experiments():
            name = str(experiment.name)
            if not name.startswith(root):
                continue
            tail = name.removeprefix(root)
            if not tail or "/" in tail or tail == "overview":
                continue
            ids.append(tail)
        return sorted(ids)
    except Exception as exc:
        raise StorageError("Failed to list MLflow experiments") from exc


def list_route_experiment_names() -> list[str]:
    """Full experiment names directly under the bound project route.

    Unlike :func:`list_experiments`, this is for cleanup: it **includes** the
    ``overview`` experiment and **soft-deleted** experiments, so a project
    delete can cascade over everything in the container. The route prefix
    naturally excludes other namespace/dry_run containers.
    """
    from mlflow.entities import ViewType

    try:
        root = f"{_routing.project_route()}/"
        names: list[str] = []
        for experiment in client.raw().search_experiments(view_type=ViewType.ALL):
            name = str(experiment.name)
            if not name.startswith(root):
                continue
            tail = name.removeprefix(root)
            if not tail or "/" in tail:
                continue
            names.append(name)
        return sorted(names)
    except Exception as exc:
        raise StorageError("Failed to list MLflow experiments for cleanup") from exc


def list_all_experiment_names() -> list[str]:
    """Every experiment name in the tracking store, including soft-deleted ones.

    Cross-project / cross-namespace; used by maintenance sweeps (e.g. QA cleanup)
    that select experiments by route convention rather than by project route.
    """
    from mlflow.entities import ViewType

    try:
        return sorted(
            str(experiment.name)
            for experiment in client.raw().search_experiments(view_type=ViewType.ALL)
        )
    except Exception as exc:
        raise StorageError("Failed to list all MLflow experiments") from exc


def _ensure_overview_run_id() -> str:
    ensure_overview()
    run = _overview_run()
    if run is None:
        raise StorageError("project overview run was not created")
    return str(run.info.run_id)


def _ensure_overview_experiment() -> str:
    name = _routing.experiment_route("overview")
    try:
        mlflow_client = client.raw()
        existing = mlflow_client.get_experiment_by_name(name)
        if existing is not None:
            return str(existing.experiment_id)
        return str(mlflow_client.create_experiment(name))
    except Exception as exc:
        raise StorageError("Failed to ensure project overview experiment") from exc


def _overview_run():
    experiment_id = _overview_experiment_id()
    if experiment_id is None:
        return None
    try:
        runs = client.raw().search_runs(
            [experiment_id],
            filter_string=f"tags.{tags.RUN_KIND} = 'project_overview'",
            max_results=1,
        )
    except Exception as exc:
        raise StorageError("Failed to read project overview run") from exc
    return list(runs)[0] if runs else None


def _overview_experiment_id() -> str | None:
    try:
        experiment = client.raw().get_experiment_by_name(_routing.experiment_route("overview"))
    except Exception as exc:
        raise StorageError("Failed to resolve project overview experiment") from exc
    if experiment is None:
        return None
    return str(experiment.experiment_id)


def _overview_from_run(run) -> ProjectOverview:
    run_tags = run.data.tags
    return ProjectOverview(
        project_name=run_tags.get(tags.PROJECT_NAME, client.bound().project_name),
        created_at=run_tags.get(tags.CREATED_AT, ""),
        current_experiment_id=run_tags.get("project.current_experiment_id") or None,
        dataset_count=int(run_tags.get("data.dataset_count", "0") or 0),
    )

__all__ = [
    "ensure_overview",
    "list_experiments",
    "read_overview",
    "write_overview",
]
