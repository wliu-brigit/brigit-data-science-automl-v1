"""Required preprocessing contract for project-mandated transformers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from sklearn.base import clone

from automl.errors import ProjectError
from automl.project import session as active_project_session


@runtime_checkable
class SklearnTransformer(Protocol):
    def fit(self, X, y=None): ...

    def transform(self, X): ...


@dataclass(frozen=True)
class RequiredTransformer:
    name: str
    transformer: SklearnTransformer
    input_cols: Sequence[str]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("RequiredTransformer.name must be a non-empty string")
        if not isinstance(self.transformer, SklearnTransformer):
            raise TypeError("RequiredTransformer.transformer must expose fit() and transform()")
        if isinstance(self.input_cols, str):
            raise TypeError("RequiredTransformer.input_cols must be a sequence of strings")
        columns = tuple(self.input_cols)
        if not columns or any(not isinstance(column, str) or not column.strip() for column in columns):
            raise ValueError("RequiredTransformer.input_cols must contain non-empty strings")
        object.__setattr__(self, "input_cols", columns)


def required_transformer_entries(session: Any | None = None) -> list[tuple[str, Any, list[str]]]:
    return [
        (requirement.name, clone(requirement.transformer), list(requirement.input_cols))
        for requirement in _requirements(session)
    ]


def describe_required_transformers(session: Any | None = None) -> list[dict[str, Any]]:
    descriptions: list[dict[str, Any]] = []
    for requirement in _requirements(session):
        transformer_type = type(requirement.transformer)
        descriptions.append(
            {
                "name": requirement.name,
                "type": transformer_type.__name__,
                "import_path": f"{transformer_type.__module__}.{transformer_type.__qualname__}",
                "columns": list(requirement.input_cols),
            }
        )
    return descriptions


def _requirements(session: Any | None) -> list[RequiredTransformer]:
    active = session
    if active is None:
        try:
            active = active_project_session()
        except ProjectError:
            return []
    return list(getattr(active.config, "required_transformers", None) or [])


__all__ = [
    "RequiredTransformer",
    "SklearnTransformer",
    "describe_required_transformers",
    "required_transformer_entries",
]
