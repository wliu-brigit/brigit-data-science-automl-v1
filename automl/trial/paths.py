"""Trial draft path helpers."""

from __future__ import annotations

import re
from pathlib import Path

from automl.mlflow import routing as mlflow_routing
from automl.project import Session


SAFE_SLUG_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def trial_slug(value: object) -> str:
    """Return a filesystem/tag-safe slug for a model or class."""

    raw = getattr(value, "name", None) or getattr(value, "__name__", None) or value.__class__.__name__
    slug = SAFE_SLUG_RE.sub("_", str(raw).strip().lower()).strip("._-")
    return slug or "trial"


def route_root(session: Session) -> Path:
    """Mode-segregated local trial route root."""

    return mlflow_routing.experiment_local_path(
        session.config.project_dir,
        project_name=session.project_name,
        experiment_id=session.active_experiment_id,
        namespace=session.namespace,
        dry_run=session.dry_run,
    )


def trial_dir(session: Session, slug: str) -> Path:
    return route_root(session) / slug


def verify_trial_dir(session: Session, path: str | Path) -> Path:
    resolved = Path(path).resolve()
    root = route_root(session).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"trial directory {resolved} is outside trial route root {root}")
    return resolved


__all__ = ["route_root", "trial_dir", "trial_slug", "verify_trial_dir"]
