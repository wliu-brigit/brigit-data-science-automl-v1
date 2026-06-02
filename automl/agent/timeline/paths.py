"""Path helpers for agent timeline artifacts."""

from __future__ import annotations

import re
from pathlib import Path

from automl.mlflow import routing as mlflow_routing
from automl.project import Session


ROUTE_SEGMENT_RE = re.compile(r"[^A-Za-z0-9_.=-]+")


def _route(active: Session) -> str:
    return mlflow_routing.experiment_route_for(
        project_name=active.project_name,
        experiment_id=active.active_experiment_id,
        namespace=active.namespace,
        dry_run=active.dry_run,
    )


def _route_segment(value: str) -> str:
    segment = ROUTE_SEGMENT_RE.sub("_", value.strip()).strip("._")
    if not segment:
        raise ValueError("route segment required")
    return segment


def _timeline_dir(project_root: Path, route: str) -> Path:
    path = project_root / ".cache" / "automl" / "tmp" / "timelines"
    for segment in [_route_segment(item) for item in route.split("/") if item] or ["unrouted"]:
        path /= segment
    return path


def _timeline_path(project_root: Path, route: str) -> Path:
    return _timeline_dir(project_root, route) / "agent_timeline.jsonl"


def _session_dir(project_root: Path, route: str, session_id: str) -> Path:
    return _timeline_dir(project_root, route) / "sessions" / _route_segment(session_id)


def _trial_dir(
    project_root: Path,
    route: str,
    session_id: str,
    trial_id: str,
    run_id: str,
) -> Path:
    return (
        _session_dir(project_root, route, session_id)
        / "trials"
        / _route_segment(trial_id)
        / _route_segment(run_id)
    )
