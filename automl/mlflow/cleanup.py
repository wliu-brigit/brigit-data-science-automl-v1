"""MLflow administrative purge for archived cleanup routes."""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

from automl.errors import ProjectError
from automl.mlflow import client as mlflow_client
from automl.mlflow import project as mlflow_project
from automl.mlflow import routing as mlflow_routing
from automl.project.session import Session, session as active_project_session
from automl.utils.io import gcs


PurgeScope = Literal["qa", "deleted"]


@dataclass(frozen=True)
class PurgePlan:
    schema_version: int = 1
    scope: str = ""
    identifier: str = ""
    project_name: str = ""
    namespace: str = ""
    dry_run: bool = False
    mlflow_experiment_targets: list[tuple[str, str]] = field(default_factory=list)
    gcs_prefix_patterns: list[str] = field(default_factory=list)
    local_paths: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PurgePlan":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            scope=str(payload.get("scope", "")),
            identifier=str(payload.get("identifier", "")),
            project_name=str(payload.get("project_name", "")),
            namespace=str(payload.get("namespace", "")),
            dry_run=bool(payload.get("dry_run", False)),
            mlflow_experiment_targets=[
                (str(item[0]), str(item[1]) if len(item) > 1 else "")
                for item in payload.get("mlflow_experiment_targets", ())
            ],
            gcs_prefix_patterns=[str(item) for item in payload.get("gcs_prefix_patterns", ())],
            local_paths=[str(item) for item in payload.get("local_paths", ())],
        )


@dataclass(frozen=True)
class PurgeResult:
    schema_version: int = 1
    mlflow_experiments: dict[str, str] = field(default_factory=dict)
    gcs: dict[str, int | str] = field(default_factory=dict)
    local: dict[str, str] = field(default_factory=dict)
    mlflow_gc_status: str = ""
    mlflow_gc_output: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PurgeResult":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            mlflow_experiments=dict(payload.get("mlflow_experiments", {})),
            gcs=dict(payload.get("gcs", {})),
            local=dict(payload.get("local", {})),
            mlflow_gc_status=str(payload.get("mlflow_gc_status", "")),
            mlflow_gc_output=str(payload.get("mlflow_gc_output", "")),
        )


@dataclass(frozen=True)
class PurgeReport:
    schema_version: int = 1
    applied: bool = False
    plan: PurgePlan = field(default_factory=PurgePlan)
    result: PurgeResult | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PurgeReport":
        result = payload.get("result")
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            applied=bool(payload.get("applied", False)),
            plan=PurgePlan.from_dict(payload.get("plan", {})),
            result=None if result is None else PurgeResult.from_dict(result),
        )


def purge(
    name: str | None = None,
    *,
    scope: PurgeScope | None = None,
    apply: bool = False,
    backend_store_uri: str = "",
    artifacts_destination: str = "",
    session: Session | None = None,
) -> PurgeReport:
    """Permanently purge archived MLflow routes and their storage artifacts."""
    active = session if session is not None else active_project_session()
    with mlflow_client.bound_for(active):
        plan = _build_plan(name=name, scope=scope, session=active)
        result = (
            _apply_plan(
                plan,
                backend_store_uri=backend_store_uri,
                artifacts_destination=artifacts_destination,
                session=active,
            )
            if apply
            else None
        )
    return PurgeReport(applied=apply, plan=plan, result=result)


def _build_plan(
    *,
    name: str | None,
    scope: PurgeScope | None,
    session: Session,
) -> PurgePlan:
    if name is None and scope is None:
        raise ValueError("purge requires name or scope")
    if name is not None and scope is not None:
        raise ValueError("purge name and scope are mutually exclusive")
    names = _all_experiment_names()
    if name is not None:
        route = name.strip("/")
        if not route.startswith("deleted/"):
            raise ValueError("purge name must start with 'deleted/'")
        targets = [item for item in names if item == route or item.startswith(route + "/")]
        return PurgePlan(
            scope="name",
            identifier=route,
            project_name=session.project_name,
            namespace=session.namespace,
            dry_run=session.dry_run,
            mlflow_experiment_targets=[(item, "") for item in targets],
            gcs_prefix_patterns=[_gcs_prefix(session, route)],
            local_paths=[str(_local_route_root(session, route))],
        )
    if scope == "qa":
        targets = [item for item in names if _is_archived_qa_namespace(item)]
        roots = sorted({_archived_namespace_root_of(item) for item in targets})
        return PurgePlan(
            scope=scope,
            identifier="qa",
            project_name=session.project_name,
            namespace=session.namespace,
            dry_run=session.dry_run,
            mlflow_experiment_targets=[(item, "") for item in targets],
            gcs_prefix_patterns=[_gcs_prefix(session, root) for root in roots],
            local_paths=[str(_local_route_root(session, root)) for root in roots],
        )
    if scope == "deleted":
        targets = [item for item in names if item.startswith("deleted/")]
        return PurgePlan(
            scope=scope,
            identifier="deleted",
            project_name=session.project_name,
            namespace=session.namespace,
            dry_run=session.dry_run,
            mlflow_experiment_targets=[(item, "") for item in targets],
            gcs_prefix_patterns=[_gcs_prefix(session, "deleted")],
            local_paths=[str(_local_route_root(session, "deleted"))],
        )
    raise ValueError(f"unknown purge scope {scope!r}")


def _apply_plan(
    plan: PurgePlan,
    *,
    backend_store_uri: str,
    artifacts_destination: str,
    session: Session,
) -> PurgeResult:
    mlflow_experiments: dict[str, str] = {}
    experiment_ids: list[str] = []
    for name, known_id in plan.mlflow_experiment_targets:
        try:
            found = mlflow_client.get_experiment_by_name(name)
            if found is None:
                mlflow_experiments[name] = "skipped: not found"
                continue
            experiment_id = known_id or str(found.experiment_id)
            if getattr(found, "lifecycle_stage", "active") != "deleted":
                mlflow_client.delete_experiment(experiment_id)
            mlflow_experiments[name] = "purged"
            experiment_ids.append(experiment_id)
        except Exception as exc:
            mlflow_experiments[name] = f"failed: {exc}"

    gcs_results: dict[str, int | str] = {}
    for prefix in plan.gcs_prefix_patterns:
        try:
            gcs_results[prefix] = gcs.delete_prefix(prefix)
        except Exception as exc:
            gcs_results[prefix] = f"failed: {exc}"
    local_results = {path: _delete_local_path(path) for path in plan.local_paths}

    gc_status, gc_output = _run_mlflow_gc(
        experiment_ids=experiment_ids,
        run_ids=[],
        backend_store_uri=backend_store_uri,
        artifacts_destination=artifacts_destination,
        session=session,
    )
    return PurgeResult(
        mlflow_experiments=mlflow_experiments,
        gcs=gcs_results,
        local=local_results,
        mlflow_gc_status=gc_status,
        mlflow_gc_output=gc_output,
    )


def _run_mlflow_gc(
    *,
    experiment_ids: list[str],
    run_ids: list[str],
    backend_store_uri: str,
    artifacts_destination: str,
    session: Session,
) -> tuple[str, str]:
    if not experiment_ids and not run_ids:
        return "skipped: no MLflow purge targets", ""
    backend_uri = _backend_store_uri(session, backend_store_uri)
    if not backend_uri:
        raise ProjectError("MLflow purge requires --backend-store-uri for remote stores")
    command = [
        "uv",
        "run",
        "mlflow",
        "gc",
        "--backend-store-uri",
        backend_uri,
    ]
    if run_ids:
        command.extend(["--run-ids", ",".join(run_ids)])
    if experiment_ids:
        command.extend(["--experiment-ids", ",".join(experiment_ids)])
    if artifacts_destination:
        command.extend(["--artifacts-destination", artifacts_destination])
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    if completed.returncode == 0:
        auth_cleanup = _prune_orphaned_auth_permissions(backend_uri)
        if auth_cleanup:
            output = "\n".join(part for part in (output, auth_cleanup) if part)
    return ("success" if completed.returncode == 0 else f"failed: {completed.returncode}", output)


def _backend_store_uri(session: Session, backend_store_uri: str) -> str:
    if backend_store_uri:
        return backend_store_uri
    tracking_uri = session.config.mlflow_tracking_uri
    if tracking_uri.startswith("file:"):
        return tracking_uri
    for candidate in (
        session.config.repo_root / "mlflow_local" / "mlflow.db",
        session.config.repo_root.parent / "mlflow_local" / "mlflow.db",
    ):
        if candidate.is_file():
            return f"sqlite:///{candidate.resolve()}"
    return ""


def _prune_orphaned_auth_permissions(backend_store_uri: str) -> str:
    tracking_db = _sqlite_path_from_uri(backend_store_uri)
    if tracking_db is None:
        return ""
    auth_db = tracking_db.with_name("basic_auth.db")
    if not auth_db.is_file():
        return ""
    try:
        with sqlite3.connect(auth_db) as db:
            db.execute(f"attach database {str(tracking_db)!r} as tracking")
            experiment_count = _delete_orphaned_auth_rows(
                db,
                table="experiment_permissions",
                key="experiment_id",
                tracking_table="experiments",
            )
            model_count = _delete_orphaned_auth_rows(
                db,
                table="registered_model_permissions",
                key="name",
                tracking_table="registered_models",
            )
            return (
                "auth permissions pruned: "
                f"experiments={experiment_count}, registered_models={model_count}"
            )
    except sqlite3.Error as exc:
        return f"auth permissions prune failed: {exc}"


def _delete_orphaned_auth_rows(
    db: sqlite3.Connection,
    *,
    table: str,
    key: str,
    tracking_table: str,
) -> int:
    if not _sqlite_table_exists(db, "main", table) or not _sqlite_table_exists(
        db, "tracking", tracking_table
    ):
        return 0
    db.execute(
        f"""
        delete from {table}
        where not exists (
            select 1
            from tracking.{tracking_table} existing
            where cast(existing.{key} as text) = cast({table}.{key} as text)
        )
        """
    )
    return int(db.execute("select changes()").fetchone()[0])


def _sqlite_table_exists(db: sqlite3.Connection, schema: str, table: str) -> bool:
    return (
        db.execute(
            f"select 1 from {schema}.sqlite_master where type = 'table' and name = ?",
            (table,),
        ).fetchone()
        is not None
    )


def _sqlite_path_from_uri(uri: str) -> Path | None:
    if not uri.startswith("sqlite:///"):
        return None
    return Path(uri.removeprefix("sqlite:///")).expanduser()


def _delete_local_path(path: str) -> str:
    target = Path(path)
    if not target.exists():
        return "skipped: not found"
    try:
        shutil.rmtree(target)
        return "deleted"
    except Exception as exc:
        return f"failed: {exc}"


def _all_experiment_names() -> list[str]:
    try:
        return mlflow_project.list_all_experiment_names()
    except Exception:
        return []


def _is_archived_qa_namespace(name: str) -> bool:
    parts = name.split("/")
    return len(parts) >= 2 and parts[0] == "deleted" and (
        parts[1] == "qa" or parts[1].startswith("qa-")
    )


def _archived_namespace_root_of(name: str) -> str:
    parts = name.split("/")
    return "/".join(parts[:3] if len(parts) >= 3 and parts[1] == "qa" else parts[:2])


def _gcs_prefix(session: Session, route: str) -> str:
    return mlflow_routing.gcs_uri_for_route(
        bucket=session.config.gcs_bucket,
        gcs_prefix=session.config.gcs_prefix,
        route=route,
    )


def _local_route_root(session: Session, route: str) -> Path:
    return session.config.project_dir / "experiments" / Path(route)


__all__ = [
    "PurgePlan",
    "PurgeReport",
    "PurgeResult",
    "purge",
]
