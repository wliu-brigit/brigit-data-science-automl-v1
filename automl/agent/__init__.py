"""Agent domain public API.

Keep this facade lazy so importing internal agent leaves such as
``automl.agent.run_options`` does not pull launch/runtime dependencies into
lightweight skill preflight scripts.
"""

from importlib import import_module
from typing import Any


_LAZY_EXPORTS = {
    "DISALLOWED": ("automl.agent.proposal", "DISALLOWED"),
    "Proposal": ("automl.agent.proposal", "Proposal"),
    "build_launch": ("automl.agent.launch", "build_launch"),
    "gather_proposer_context": (
        "automl.agent.proposer_context",
        "gather_proposer_context",
    ),
    "handle_event": ("automl.agent.timeline", "handle_event"),
    "publish": ("automl.agent.timeline", "publish"),
    "validate_proposal": ("automl.agent.checks", "validate_proposal"),
}

__all__ = [
    "DISALLOWED",
    "Proposal",
    "build_launch",
    "gather_proposer_context",
    "handle_event",
    "publish",
    "validate_proposal",
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
