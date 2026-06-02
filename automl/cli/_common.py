"""Shared CLI helpers."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any

from automl.project import use_project


def session_from_args(args: Any, *, experiment_id: str | None = None):
    return use_project(
        args.project or None,
        repo_root=args.project_root,
        dry_run=bool(args.dry_run),
        namespace=args.namespace or "",
        experiment_id=experiment_id or args.experiment_id,
    )


def jsonable(value: Any) -> Any:
    if hasattr(value, "to_json"):
        return value.to_json()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def print_json(value: Any) -> None:
    print(json.dumps(jsonable(value), indent=2, default=str))


__all__ = ["jsonable", "print_json", "session_from_args"]
