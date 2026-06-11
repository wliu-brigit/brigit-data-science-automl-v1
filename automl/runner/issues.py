"""Trial issue ledger: the durable record of what went wrong mid-trial.

Best-effort steps record here instead of swallowing exceptions. Events are
appended to a local JSONL as they happen (crash-safe: a native crash of the
runner still leaves the file on disk) and published to MLflow at trial end by
``automl.runner.issue_artifacts``. Recording must never raise — the ledger
cannot be allowed to become a failure source itself.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from automl.runner.failures import ExceptionSnapshot


class IssueRecorder:
    def __init__(self, jsonl_path: Path | None = None) -> None:
        self._issues: list[dict[str, Any]] = []
        self._jsonl_path = jsonl_path

    def record(
        self,
        problem: BaseException | str,
        *,
        phase: str,
        severity: str = "error",
    ) -> None:
        if isinstance(problem, BaseException):
            snapshot = ExceptionSnapshot.from_exception(problem)
            error_class = snapshot.error_class
            message = snapshot.message
            traceback_tail = snapshot.to_dict()["traceback_tail"]
        else:
            error_class = ""
            message = str(problem)
            traceback_tail = []
        entry = {
            "at": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "severity": severity,
            "error_class": error_class,
            "message": message,
            "traceback_tail": traceback_tail,
        }
        self._issues.append(entry)
        self._append_jsonl(entry)

    def snapshot(self) -> list[dict[str, Any]]:
        return [dict(entry) for entry in self._issues]

    @property
    def count(self) -> int:
        return len(self._issues)

    def _append_jsonl(self, entry: dict[str, Any]) -> None:
        if self._jsonl_path is None:
            return
        try:
            with self._jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
        except OSError:
            # Ledger writes are best-effort by definition; the in-memory
            # record still publishes at trial end.
            pass


__all__ = ["IssueRecorder"]
