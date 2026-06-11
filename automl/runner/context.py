"""Trial-scoped context: one home for the runner's cross-cutting state.

Composes the trial's identity (filled in as it becomes known), the
``TimingRecorder``, and the ``IssueRecorder`` so one object threads through
the runner instead of N loose parameters. This is the *record* side of the
record/machinery split (design: trial-reliability §7) — a future step-based
runner passes this same context to each step.
"""

from __future__ import annotations

from pathlib import Path

from automl.runner.issues import IssueRecorder
from automl.runner.timing import TimingRecorder

ISSUES_JSONL_NAME = "issues.jsonl"


class TrialContext:
    def __init__(self, *, trial_dir: Path | None = None) -> None:
        self.run_id: str = ""
        self.trial_id: str = ""
        self.trial_number: int | None = None
        self.slug: str = ""
        self.strategy: str = ""
        self.trial_dir = trial_dir
        self.timing = TimingRecorder()
        # Memory-only when there is no trial dir to leave evidence in
        # (project-run trials); the published record still lands either way.
        self.issues = IssueRecorder(
            jsonl_path=(trial_dir / ISSUES_JSONL_NAME) if trial_dir else None
        )

    def phase(self, name: str):
        return self.timing.phase(name)

    def record_issue(
        self,
        problem: BaseException | str,
        *,
        phase: str | None = None,
        severity: str = "error",
    ) -> None:
        resolved = phase or self.timing.last_phase or "unknown"
        self.issues.record(problem, phase=resolved, severity=severity)


__all__ = ["ISSUES_JSONL_NAME", "TrialContext"]
