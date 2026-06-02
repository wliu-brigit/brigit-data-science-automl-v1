"""Project dependency allow-list helpers."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from automl.project.session import session as active_project_session

if TYPE_CHECKING:
    from automl.project.session import Session


PACKAGE_NAME_RE = re.compile(r"^([A-Za-z0-9._-]+)")


def parse_dependency_name(spec: str) -> str:
    raw = spec.split("[", 1)[0].strip()
    match = PACKAGE_NAME_RE.match(raw)
    return match.group(1) if match else raw


def allowed_dependencies(session: Session | None = None) -> list[str]:
    active = session if session is not None else active_project_session()
    path = Path(active.config.repo_root) / "pyproject.toml"
    if not path.exists():
        return []
    with path.open("rb") as handle:
        payload = tomllib.load(handle)

    names: list[str] = []
    seen: set[str] = set()
    for dep in _dependency_strings(payload):
        name = parse_dependency_name(dep)
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _dependency_strings(payload: dict[str, Any]) -> list[str]:
    deps: list[str] = []
    project = payload.get("project")
    if isinstance(project, dict):
        deps.extend(item for item in project.get("dependencies", []) if isinstance(item, str))
    groups = payload.get("dependency-groups")
    if isinstance(groups, dict):
        for group_deps in groups.values():
            if isinstance(group_deps, list):
                deps.extend(item for item in group_deps if isinstance(item, str))
    return deps


__all__ = ["allowed_dependencies", "parse_dependency_name"]
