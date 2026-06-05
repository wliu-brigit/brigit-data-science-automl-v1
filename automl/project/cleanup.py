"""Project-owned cleanup cascade for MLflow, GCS, and local route artifacts."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Mapping

from automl.mlflow import client as mlflow_client
from automl.mlflow import project as mlflow_project
from automl.mlflow import routing as mlflow_routing
from automl.project.session import Session, session as active_project_session
from automl.trial.types import ParentExperimentRef
from automl.utils.io import gcs


CleanupScope = Literal["project", "experiment", "trial", "qa"]

# QA is a transient namespace convention: new throwaway test runs route under
# ``qa/<purpose>``. The older ``qa-*`` prefix remains sweepable for cleanup.
QA_NAMESPACE_PREFIXES = ("qa-", "qa/")


@dataclass(frozen=True)
class CleanupPlan:
    schema_version: int = 1
    scope: str = "project"
    identifier: str = ""
    project_name: str = ""
    namespace: str = ""
    dry_run: bool = False
    mlflow_experiment_targets: list[tuple[str, str]] = field(default_factory=list)
    mlflow_run_targets: list[str] = field(default_factory=list)
    gcs_prefix_patterns: list[str] = field(default_factory=list)
    local_paths: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CleanupPlan":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            scope=str(payload.get("scope", "project")),
            identifier=str(payload.get("identifier", "")),
            project_name=str(payload.get("project_name", "")),
            namespace=str(payload.get("namespace", "")),
            dry_run=bool(payload.get("dry_run", False)),
            mlflow_experiment_targets=[
                (str(item[0]), str(item[1]) if len(item) > 1 else "")
                for item in payload.get("mlflow_experiment_targets", ())
            ],
            mlflow_run_targets=[str(item) for item in payload.get("mlflow_run_targets", ())],
            gcs_prefix_patterns=[str(item) for item in payload.get("gcs_prefix_patterns", ())],
            local_paths=[str(item) for item in payload.get("local_paths", ())],
        )


@dataclass(frozen=True)
class CleanupResult:
    schema_version: int = 1
    mlflow_experiments: dict[str, str] = field(default_factory=dict)
    mlflow_runs: dict[str, str] = field(default_factory=dict)
    gcs: dict[str, int | str] = field(default_factory=dict)
    local: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CleanupResult":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            mlflow_experiments=dict(payload.get("mlflow_experiments", {})),
            mlflow_runs=dict(payload.get("mlflow_runs", {})),
            gcs=dict(payload.get("gcs", {})),
            local=dict(payload.get("local", {})),
        )


@dataclass(frozen=True)
class CleanupReport:
    schema_version: int = 1
    applied: bool = False
    plan: CleanupPlan = field(default_factory=CleanupPlan)
    result: CleanupResult | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CleanupReport":
        result = payload.get("result")
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            applied=bool(payload.get("applied", False)),
            plan=CleanupPlan.from_dict(payload.get("plan", {})),
            result=None if result is None else CleanupResult.from_dict(result),
        )


def delete(
    name: str | None = None,
    *,
    scope: CleanupScope = "project",
    apply: bool = False,
    session: Session | None = None,
    parent_experiment: ParentExperimentRef | None = None,
) -> CleanupReport:
    active = session if session is not None else active_project_session()
    with mlflow_client.bound_for(active):
        plan = _build_plan(scope, name, active, parent_experiment=parent_experiment)
        result = (
            _apply_plan(
                plan,
                session=active,
            )
            if apply
            else None
        )
    return CleanupReport(applied=apply, plan=plan, result=result)


def _build_plan(
    scope: str,
    identifier: str | None,
    session: Session,
    *,
    parent_experiment: ParentExperimentRef | None = None,
) -> CleanupPlan:
    if scope == "qa":
        if identifier is not None:
            raise ValueError("qa cleanup scope does not accept name")
        names = [name for name in _all_experiment_names() if _is_qa_namespace(name)]
        namespaces = sorted({_namespace_root_of(name) for name in names})
        return CleanupPlan(
            scope=scope,
            identifier="qa",
            project_name=session.project_name,
            namespace=session.namespace,
            dry_run=session.dry_run,
            mlflow_experiment_targets=[(name, "") for name in names],
            gcs_prefix_patterns=[_namespace_gcs_prefix(session, ns) for ns in namespaces],
            local_paths=[str(_namespace_local_root(session, ns)) for ns in namespaces],
        )
    if identifier is None:
        raise ValueError(f"{scope} cleanup requires name")
    if scope == "experiment":
        route = _route_for(session, identifier)
        return CleanupPlan(
            scope=scope,
            identifier=identifier,
            project_name=session.project_name,
            namespace=session.namespace,
            dry_run=session.dry_run,
            mlflow_experiment_targets=[(route, "")],
            gcs_prefix_patterns=[_gcs_prefix(session, route)],
            local_paths=[str(_local_route_root(session, identifier))],
        )
    if scope == "project":
        if identifier != session.project_name:
            raise ValueError(
                f"project cleanup name {identifier!r} does not match active project "
                f"{session.project_name!r}"
            )
        names = _route_experiment_names()
        if not names:
            names = [_project_route(session)]
        # Local route artifacts are container-scoped; the project-level
        # ``.cache/automl`` scratch (locks/timelines) is shared across
        # dry_run/namespace, so only the canonical base cleanup may remove it.
        local_paths = [str(_project_local_root(session))]
        if not session.dry_run and not session.namespace:
            local_paths.append(str(session.config.project_dir / ".cache" / "automl"))
        return CleanupPlan(
            scope=scope,
            identifier=identifier,
            project_name=session.project_name,
            namespace=session.namespace,
            dry_run=session.dry_run,
            mlflow_experiment_targets=[(name, "") for name in names],
            gcs_prefix_patterns=[_project_gcs_prefix(session)],
            local_paths=local_paths,
        )
    if scope == "trial":
        if parent_experiment is None:
            raise ValueError("trial cleanup requires parent_experiment")
        route = parent_experiment.mlflow_experiment_name
        return CleanupPlan(
            scope=scope,
            identifier=identifier,
            project_name=session.project_name,
            namespace=session.namespace,
            dry_run=session.dry_run,
            mlflow_experiment_targets=[],
            mlflow_run_targets=[identifier],
            gcs_prefix_patterns=[_trial_gcs_prefix(session, route, identifier)],
            local_paths=[str(_local_route_root(session, parent_experiment.experiment_id) / identifier)],
        )
    raise ValueError(f"unknown cleanup scope {scope!r}")


def _apply_plan(
    plan: CleanupPlan,
    *,
    session: Session,
) -> CleanupResult:
    mlflow_experiments: dict[str, str] = {}
    mlflow_runs: dict[str, str] = {}
    archive_root = _archive_root()

    for name, known_id in plan.mlflow_experiment_targets:
        try:
            found = mlflow_client.get_experiment_by_name(name)
            if found is None:
                mlflow_experiments[name] = "skipped: not found"
                continue
            experiment_id = str(found.experiment_id)
            if getattr(found, "lifecycle_stage", "active") == "deleted":
                mlflow_experiments[name] = "skipped: already deleted"
            else:
                archived_name = _available_archive_experiment_name(name, archive_root)
                mlflow_client.rename_experiment(experiment_id, archived_name)
                mlflow_client.delete_experiment(experiment_id)
                mlflow_experiments[name] = f"archived: {archived_name}; deleted"
        except Exception as exc:
            mlflow_experiments[name] = f"failed: {exc}"

    for run_id in plan.mlflow_run_targets:
        try:
            mlflow_client.delete_run(run_id)
            mlflow_runs[run_id] = "deleted"
        except Exception as exc:
            mlflow_runs[run_id] = f"failed: {exc}"

    gcs_results: dict[str, int | str] = {}
    for prefix in plan.gcs_prefix_patterns:
        try:
            archived_prefix = _archive_gcs_prefix(prefix, archive_root)
            moved = gcs.move_prefix(prefix, archived_prefix)
            gcs_results[prefix] = f"archived {moved} to {archived_prefix}"
        except Exception as exc:
            gcs_results[prefix] = f"failed: {exc}"
    local_results = {
        path: _archive_local_path(path, archive_root)
        for path in plan.local_paths
    }

    return CleanupResult(
        mlflow_experiments=mlflow_experiments,
        mlflow_runs=mlflow_runs,
        gcs=gcs_results,
        local=local_results,
    )


def _archive_root() -> str:
    return "deleted"


def _available_archive_experiment_name(name: str, archive_root: str) -> str:
    candidate = f"{archive_root}/{name}".strip("/")
    if mlflow_client.get_experiment_by_name(candidate) is None:
        return candidate
    index = 2
    while mlflow_client.get_experiment_by_name(f"{candidate}_{index}") is not None:
        index += 1
    return f"{candidate}_{index}"


def _archive_gcs_prefix(prefix: str, archive_root: str) -> str:
    bucket, blob_prefix = gcs.parse_gcs_uri(prefix.rstrip("/") + "/_")
    blob_prefix = blob_prefix.removesuffix("_").strip("/")
    configured_prefix = mlflow_client.bound().gcs_prefix.strip("/")
    if configured_prefix and blob_prefix.startswith(configured_prefix + "/"):
        route = blob_prefix.removeprefix(configured_prefix + "/")
        archived_blob_prefix = f"{configured_prefix}/{archive_root}/{route}"
    else:
        archived_blob_prefix = f"{archive_root}/{blob_prefix}"
    return f"gs://{bucket}/{archived_blob_prefix.strip('/')}/"


def _delete_local_path(path: str) -> str:
    target = Path(path)
    if not target.exists():
        return "skipped: not found"
    try:
        shutil.rmtree(target)
        return "deleted"
    except Exception as exc:
        return f"failed: {exc}"


def _archive_local_path(path: str, archive_root: str) -> str:
    target = Path(path)
    if not target.exists():
        return "skipped: not found"
    parts = target.parts
    if "experiments" not in parts:
        return _delete_local_path(path)
    index = parts.index("experiments")
    route_parts = parts[index + 1 :]
    destination = Path(*parts[: index + 1], *Path(archive_root).parts, *route_parts)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination = _available_local_archive_path(destination)
        shutil.move(str(target), str(destination))
        return f"archived: {destination}"
    except Exception as exc:
        return f"failed: {exc}"


def _available_local_archive_path(path: Path) -> Path:
    index = 2
    candidate = path
    while candidate.exists():
        candidate = path.with_name(f"{path.name}_{index}")
        index += 1
    return candidate


def _route_experiment_names() -> list[str]:
    """Full experiment names under the bound route, incl. overview + soft-deleted."""
    try:
        return mlflow_project.list_route_experiment_names()
    except Exception:
        return []


def _all_experiment_names() -> list[str]:
    """Every experiment name in the store (cross-project), incl. soft-deleted."""
    try:
        return mlflow_project.list_all_experiment_names()
    except Exception:
        return []


def _is_qa_namespace(name: str) -> bool:
    return name.startswith(QA_NAMESPACE_PREFIXES)


def _is_archived_qa_namespace(name: str) -> bool:
    parts = name.split("/")
    return len(parts) >= 3 and parts[0] == "deleted" and (
        parts[2] == "qa" or parts[2].startswith("qa-")
    )


def _namespace_root_of(name: str) -> str:
    if _is_archived_qa_namespace(name):
        parts = name.split("/")
        return "/".join(parts[:4] if parts[2] == "qa" else parts[:3])
    return _namespace_of(name)


def _namespace_of(name: str) -> str:
    try:
        return str(mlflow_routing.parse_experiment_route(name)["namespace"])
    except Exception:
        return name.split("/", 1)[0]


def _namespace_gcs_prefix(session: Session, namespace: str) -> str:
    return _gcs_prefix(session, namespace)


def _namespace_local_root(session: Session, namespace: str) -> Path:
    return session.config.project_dir / "experiments" / Path(namespace)


def _route_for(session: Session, experiment_id: str) -> str:
    return mlflow_routing.experiment_route_for(
        project_name=session.project_name,
        experiment_id=experiment_id,
        namespace=session.namespace,
        dry_run=session.dry_run,
    )


def _project_route(session: Session) -> str:
    return mlflow_routing.project_route_for(
        project_name=session.project_name,
        namespace=session.namespace,
        dry_run=session.dry_run,
    )


def _gcs_prefix(session: Session, route: str) -> str:
    return mlflow_routing.gcs_uri_for_route(
        bucket=session.config.gcs_bucket,
        gcs_prefix=session.config.gcs_prefix,
        route=route,
    )


def _project_gcs_prefix(session: Session) -> str:
    """Wholesale project-route prefix: catches data/, overview/, every experiment."""
    return _gcs_prefix(session, _project_route(session))


def _trial_gcs_prefix(session: Session, route: str, run_id: str) -> str:
    partition_time = datetime.now(UTC)
    try:
        start_time = mlflow_client.run_start_time(run_id)
        if start_time:
            partition_time = datetime.fromtimestamp(start_time / 1000, UTC)
    except Exception:
        pass
    return mlflow_routing.run_gcs_uri_for_route(
        bucket=session.config.gcs_bucket,
        gcs_prefix=session.config.gcs_prefix,
        route=route,
        run_id=run_id,
        now=partition_time,
    )


def _local_route_root(session: Session, experiment_id: str) -> Path:
    return mlflow_routing.experiment_local_path(
        session.config.project_dir,
        project_name=session.project_name,
        experiment_id=experiment_id,
        namespace=session.namespace,
        dry_run=session.dry_run,
    )


def _project_local_root(session: Session) -> Path:
    return (
        session.config.project_dir
        / "experiments"
        / Path(
            mlflow_routing.project_route_for(
                project_name=session.project_name,
                namespace=session.namespace,
                dry_run=session.dry_run,
            )
        )
    )


__all__ = ["CleanupPlan", "CleanupReport", "CleanupResult", "delete"]
