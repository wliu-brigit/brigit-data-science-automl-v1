"""Runner-owned failure report models."""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_TRACEBACK_TAIL_LINES = 80


@dataclass(frozen=True)
class ExceptionSnapshot:
    error_class: str
    message: str
    traceback_text: str

    @classmethod
    def from_exception(cls, exc: BaseException) -> "ExceptionSnapshot":
        return cls(
            error_class=type(exc).__name__,
            message=str(exc),
            traceback_text="".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
        )

    def to_dict(self, *, include_traceback: bool = False) -> dict[str, Any]:
        payload = {
            "error_class": self.error_class,
            "message": self.message,
            "traceback_tail": self.traceback_text.splitlines()[-_TRACEBACK_TAIL_LINES:],
        }
        if include_traceback:
            payload["traceback"] = self.traceback_text
        return payload


@dataclass(frozen=True)
class RunnerFailureReport:
    runner_kind: str
    phase: str
    exception: ExceptionSnapshot
    run_id: str = ""
    project_name: str = ""
    experiment_id: str = ""
    trial_id: str = ""
    trial_number: int | None = None
    trial_slug: str = ""
    trial_strategy: str = ""
    trial_dir: Path | None = None
    timing: dict[str, Any] = field(default_factory=dict)

    def to_dict(
        self,
        *,
        traceback_artifact: str,
        proposal_artifact: str = "",
    ) -> dict[str, Any]:
        payload = {
            "schema_version": 1,
            "status": "failed",
            "runner_kind": self.runner_kind,
            "phase": self.phase,
            "run_id": self.run_id,
            "project_name": self.project_name,
            "experiment_id": self.experiment_id,
            "trial_id": self.trial_id,
            "trial_number": int(self.trial_number or 0),
            "trial_slug": self.trial_slug,
            "trial_strategy": self.trial_strategy,
            "traceback_artifact": traceback_artifact,
            "timing": dict(self.timing),
        }
        payload.update(self.exception.to_dict())
        if self.trial_dir is not None:
            payload["trial_dir"] = str(self.trial_dir)
        if proposal_artifact:
            payload["proposal_artifact"] = proposal_artifact
        return payload


__all__ = ["ExceptionSnapshot", "RunnerFailureReport"]
