"""Runner-owned result policy helpers."""

from __future__ import annotations

from typing import Any

from automl.trial.types import TrialStatus


def trial_status_value(status: object) -> str:
    return str(getattr(status, "value", status))


def trial_result_exit_code(result: dict[str, Any] | object) -> int:
    status = result.get("status") if isinstance(result, dict) else getattr(result, "status", "")
    return 0 if trial_status_value(status) == TrialStatus.FINISHED.value else 1


__all__ = ["trial_result_exit_code", "trial_status_value"]
