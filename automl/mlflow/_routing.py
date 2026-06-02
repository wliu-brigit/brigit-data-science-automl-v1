"""Internal route and GCS URI construction for the MLflow seam."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from automl.errors import StorageError
from automl.mlflow import client


_SAFE_ROUTE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
GCSPathKind = Literal["run_bulk", "validation", "agent_events"]


def namespace_route_for(*, namespace: str = "", dry_run: bool = False) -> str:
    """Return ``[namespace/][dry_run]`` for explicit route inputs."""
    segments = [
        *_namespace_segments(namespace),
        *(["dry_run"] if dry_run else []),
    ]
    return "/".join(segments)


def namespace_route_prefix_for(
    *,
    gcs_prefix: str,
    namespace: str = "",
    dry_run: bool = False,
) -> str:
    """Return ``gcs_prefix/[namespace/][dry_run]`` for explicit route inputs."""
    root = gcs_prefix.strip("/")
    route = namespace_route_for(namespace=namespace, dry_run=dry_run)
    return "/".join(part for part in (root, route) if part)


def project_route_for(*, project_name: str, dry_run: bool = False, namespace: str = "") -> str:
    """Return ``[namespace/][dry_run/]project`` for explicit route inputs."""
    segments = [
        *_namespace_segments(namespace),
        *(["dry_run"] if dry_run else []),
        _validate_route_component("project_name", project_name),
    ]
    return "/".join(segments)


def project_route() -> str:
    """Return ``[namespace/][dry_run/]project`` for the bound session."""
    bound = client.bound()
    return project_route_for(
        project_name=bound.project_name,
        dry_run=bound.dry_run,
        namespace=bound.namespace,
    )


def experiment_route(experiment_id: str | None = None) -> str:
    """Return ``[namespace/][dry_run/]project/experiment`` for the bound session."""
    resolved_experiment_id = _resolve_experiment_id(experiment_id)
    bound = client.bound()
    return experiment_route_for(
        project_name=bound.project_name,
        experiment_id=resolved_experiment_id,
        namespace=bound.namespace,
        dry_run=bound.dry_run,
    )


def experiment_route_for(
    project_name: str,
    experiment_id: str,
    namespace: str = "",
    dry_run: bool = False,
) -> str:
    """Return ``[namespace/][dry_run/]project/experiment`` for explicit inputs."""
    return "/".join(
        (
            project_route_for(
                project_name=project_name,
                dry_run=dry_run,
                namespace=namespace,
            ),
            _validate_route_component("experiment_id", experiment_id),
        )
    )


def parse_experiment_route(route: str) -> dict[str, object]:
    """Parse supported experiment route grammar into explicit route inputs."""
    try:
        segments = _route_segments(route)
        if len(segments) < 2:
            raise ValueError("experiment route requires project and experiment segments")

        prefix = segments[:-2]
        project_name, experiment_id = segments[-2:]
        if prefix and prefix[-1] == "dry_run":
            dry_run = True
            namespace_segments = prefix[:-1]
        else:
            dry_run = False
            namespace_segments = prefix
        if "dry_run" in namespace_segments:
            raise ValueError("namespace segment cannot use reserved dry_run marker")

        return {
            "namespace": "/".join(namespace_segments),
            "dry_run": dry_run,
            "project_name": project_name,
            "experiment_id": experiment_id,
        }
    except ValueError as exc:
        raise StorageError(f"Malformed experiment route {route!r}") from exc


def experiment_local_path(
    root: Path,
    *,
    project_name: str,
    experiment_id: str,
    namespace: str = "",
    dry_run: bool = False,
) -> Path:
    """Return the local ``root/experiments/[namespace/][dry_run/]project/experiment`` path."""
    route = experiment_route_for(
        project_name=project_name,
        experiment_id=experiment_id,
        namespace=namespace,
        dry_run=dry_run,
    )
    return root / "experiments" / Path(route)


def project_route_prefix() -> str:
    bound = client.bound()
    root = bound.gcs_prefix.strip("/")
    return "/".join(part for part in (root, project_route()) if part)


def experiment_route_prefix_for(
    *,
    gcs_prefix: str,
    project_name: str,
    experiment_id: str,
    namespace: str = "",
    dry_run: bool = False,
) -> str:
    """Return ``gcs_prefix/[namespace/][dry_run/]project/experiment``."""
    root = gcs_prefix.strip("/")
    route = experiment_route_for(
        project_name=project_name,
        experiment_id=experiment_id,
        namespace=namespace,
        dry_run=dry_run,
    )
    return "/".join(part for part in (root, route) if part)


def gcs_uri_for_route(
    *,
    bucket: str,
    gcs_prefix: str,
    route: str,
) -> str:
    """Return ``gs://bucket/gcs_prefix/route/`` with route grammar centralized."""
    if not bucket:
        raise ValueError("bucket required")
    root = gcs_prefix.strip("/")
    normalized_route = "/".join(_route_segments(route))
    prefix = "/".join(part for part in (root, normalized_route) if part)
    return f"gs://{bucket}/{prefix}/"


def run_gcs_uri_for_route(
    *,
    bucket: str,
    gcs_prefix: str,
    route: str,
    run_id: str,
    now: datetime | None = None,
) -> str:
    """Return a run-scoped ``gs://`` prefix under an experiment route."""
    if not run_id:
        raise ValueError("run_id required for run route")
    route_parts = parse_experiment_route(route)
    routed_experiment = experiment_route_for(
        project_name=str(route_parts["project_name"]),
        experiment_id=str(route_parts["experiment_id"]),
        namespace=str(route_parts["namespace"]),
        dry_run=bool(route_parts["dry_run"]),
    )
    partition = (now or datetime.now(UTC)).strftime("%Y-%m")
    return gcs_uri_for_route(
        bucket=bucket,
        gcs_prefix=gcs_prefix,
        route="/".join((routed_experiment, "runs", partition, run_id)),
    )


def route_prefix_for(experiment_id: str | None = None) -> str:
    bound = client.bound()
    root = bound.gcs_prefix.strip("/")
    if not root:
        raise ValueError("gcs_prefix required")
    resolved_experiment_id = _resolve_experiment_id(experiment_id)
    return experiment_route_prefix_for(
        gcs_prefix=root,
        project_name=bound.project_name,
        experiment_id=resolved_experiment_id,
        namespace=bound.namespace,
        dry_run=bound.dry_run,
    )


def bucket_uri_for(
    *,
    kind: GCSPathKind,
    run_id: str,
    experiment_id: str | None = None,
    now: datetime | None = None,
) -> str:
    """Return a canonical ``gs://`` prefix with a trailing slash."""
    bound = client.bound()
    if not bound.bucket:
        raise ValueError("bucket required")
    if not run_id:
        raise ValueError(f"run_id required for {kind}")
    run_base = run_gcs_uri_for_route(
        bucket=bound.bucket,
        gcs_prefix=bound.gcs_prefix,
        route=experiment_route(experiment_id),
        run_id=run_id,
        now=now,
    )
    if kind == "run_bulk":
        return run_base
    if kind == "validation":
        return f"{run_base}validation/"
    if kind == "agent_events":
        return f"{run_base}agent/"
    raise ValueError(f"unknown kind={kind!r}")


def _resolve_experiment_id(experiment_id: str | None = None) -> str:
    resolved = experiment_id if experiment_id is not None else client.bound().experiment_id
    if resolved is None:
        raise ValueError("experiment_id required")
    return resolved


def _namespace_segments(namespace: str) -> list[str]:
    segments = [segment for segment in namespace.strip("/").split("/") if segment]
    validated_segments = []
    for index, segment in enumerate(segments):
        validated = _validate_route_component(f"namespace[{index}]", segment)
        if validated == "dry_run":
            raise ValueError("namespace segment cannot use reserved dry_run marker")
        validated_segments.append(validated)
    return validated_segments


def _validate_route_component(name: str, value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} required")
    if not _SAFE_ROUTE_COMPONENT_RE.fullmatch(value):
        raise ValueError(f"{name} must contain only letters, numbers, '_', '-', or '.'")
    return value


def _route_segments(route: str) -> list[str]:
    if not isinstance(route, str) or not route:
        raise ValueError("route required")
    if route != route.strip("/"):
        raise ValueError("route must not start or end with '/'")
    segments = route.split("/")
    if any(not segment for segment in segments):
        raise ValueError("route must not contain empty segments")
    return [
        _validate_route_component(f"route[{index}]", segment)
        for index, segment in enumerate(segments)
    ]


__all__ = [
    "bucket_uri_for",
    "experiment_local_path",
    "experiment_route",
    "experiment_route_for",
    "experiment_route_prefix_for",
    "gcs_uri_for_route",
    "namespace_route_for",
    "namespace_route_prefix_for",
    "parse_experiment_route",
    "project_route",
    "project_route_for",
    "project_route_prefix",
    "route_prefix_for",
    "run_gcs_uri_for_route",
]
