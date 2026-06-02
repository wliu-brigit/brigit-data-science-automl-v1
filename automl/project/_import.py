"""Private helpers for importing project-local config modules."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


PROJECTS_DIR = "projects"
PROJECT_FILENAME = "config.py"
INSTRUCTIONS_FILENAME = "PROJECT_INSTRUCTIONS.md"


def import_project_config(repo_root: Path, project_name: str) -> ModuleType:
    """Import ``projects.<name>.config`` from a specific repository root."""

    root = repo_root.resolve()
    for name, module in list(sys.modules.items()):
        if name == PROJECTS_DIR or name.startswith(f"{PROJECTS_DIR}."):
            if module is None or not _module_belongs_to_root(module, root):
                del sys.modules[name]

    root_path = str(root)
    old_path = list(sys.path)
    try:
        sys.path[:] = [root_path, *[item for item in sys.path if item != root_path]]
        importlib.invalidate_caches()
        return importlib.import_module(f"{PROJECTS_DIR}.{project_name}.config")
    finally:
        sys.path[:] = old_path


def _module_belongs_to_root(module: ModuleType, root: Path) -> bool:
    for path in _module_paths(module):
        try:
            if path.resolve().is_relative_to(root):
                return True
        except (OSError, RuntimeError):
            continue
    return False


def _module_paths(module: ModuleType) -> list[Path]:
    paths: list[Path] = []
    module_file = getattr(module, "__file__", None)
    if module_file:
        paths.append(Path(module_file))
    module_path = getattr(module, "__path__", None)
    if module_path is not None:
        paths.extend(Path(item) for item in module_path)
    spec = getattr(module, "__spec__", None)
    if spec is not None:
        origin = getattr(spec, "origin", None)
        if origin and origin not in {"built-in", "frozen", "namespace"}:
            paths.append(Path(origin))
        locations: Any = getattr(spec, "submodule_search_locations", None)
        if locations is not None:
            paths.extend(Path(item) for item in locations)
    return paths


__all__ = [
    "INSTRUCTIONS_FILENAME",
    "PROJECT_FILENAME",
    "PROJECTS_DIR",
    "import_project_config",
]
