"""Project discovery helpers for CLI/session bootstrap."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from automl.errors import ProjectError

from ._import import PROJECTS_DIR
from .config import ProjectConfig


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / PROJECTS_DIR).is_dir():
            return candidate
    raise ProjectError(f"could not find repo root containing {PROJECTS_DIR}/ from {current}")


def list_projects(*, repo_root: Path | None = None) -> list[str]:
    root = Path(repo_root).resolve() if repo_root is not None else find_repo_root()
    projects_root = root / PROJECTS_DIR
    if not projects_root.exists():
        return []
    return sorted(
        path.name
        for path in projects_root.iterdir()
        if path.is_dir() and (path / "config.py").exists()
    )


def infer_project_name(*, repo_root: Path | None = None, start: Path | None = None) -> str:
    root = Path(repo_root).resolve() if repo_root is not None else find_repo_root(start)
    cursor = (start or Path.cwd()).resolve()
    projects_root = root / PROJECTS_DIR
    try:
        relative = cursor.relative_to(projects_root)
    except ValueError:
        relative = None
    if relative is not None and relative.parts:
        candidate = relative.parts[0]
        if (projects_root / candidate / "config.py").exists():
            return candidate

    names = list_projects(repo_root=root)
    if len(names) == 1:
        return names[0]
    if not names:
        raise ProjectError(f"no projects found under {projects_root}")
    raise ProjectError(
        "multiple projects found; pass --project explicitly: " + ", ".join(names)
    )


def project_metadata(
    *,
    project: str | None = None,
    repo_root: Path | None = None,
    start: Path | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    name = project or infer_project_name(repo_root=repo_root, start=start)
    config = ProjectConfig.load(name, repo_root=repo_root)
    if strict and not config.is_complete():
        missing = ", ".join(config.missing_fields())
        raise ProjectError(f"project {name!r} is incomplete: missing {missing}")
    return {
        "project_name": config.project_name,
        "repo_root": str(config.repo_root),
        "project_dir": str(config.project_dir),
        "config_path": str(config.config_path),
        "instructions_path": str(config.instructions_path),
        "is_complete": config.is_complete(),
        "missing_fields": config.missing_fields(),
        "experiment_id": config.run_config.experiment_id if config.run_config else "",
    }


__all__ = ["find_repo_root", "infer_project_name", "list_projects", "project_metadata"]
