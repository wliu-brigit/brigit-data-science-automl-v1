"""Trial domain public API.

Keep this facade lightweight: runner imports pure trial leaves during execution,
and importing those leaves must not pull trial workflow modules into memory.
"""

from importlib import import_module
from typing import Any

from automl.trial.types import (
    ArtifactRef,
    ParentExperimentRef,
    TrialDetails,
    TrialStatus,
    TrialSummary,
)


_LAZY_EXPORTS = {
    "create": ("automl.trial.create", "create"),
    "delete": ("automl.trial.cleanup", "delete"),
    "fork": ("automl.trial.fork", "fork"),
    "load_model": ("automl.trial.show", "load_model"),
    "package_model": ("automl.trial.packaging", "package_model"),
    "show_trial": ("automl.trial.show", "show_trial"),
}

__all__ = [
    "ArtifactRef",
    "create",
    "delete",
    "fork",
    "load_model",
    "package_model",
    "ParentExperimentRef",
    "show_trial",
    "TrialDetails",
    "TrialStatus",
    "TrialSummary",
]


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
