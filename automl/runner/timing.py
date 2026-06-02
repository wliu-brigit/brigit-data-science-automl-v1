"""Runner timing helpers."""

from __future__ import annotations

import time
from contextlib import contextmanager


class TimingRecorder:
    """Collect runner phase durations in seconds for MLflow artifacts."""

    def __init__(self) -> None:
        self.started = time.monotonic()
        self.phases: dict[str, float] = {}
        self.last_phase: str | None = None

    @contextmanager
    def phase(self, name: str):
        self.last_phase = name
        started = time.monotonic()
        try:
            yield
        finally:
            self.phases[name] = self.phases.get(name, 0.0) + (
                time.monotonic() - started
            )

    def snapshot(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "unit": "seconds",
            "total_seconds": max(0.0, time.monotonic() - self.started),
            "phases": dict(self.phases),
        }


@contextmanager
def timed_phase(timing: TimingRecorder | None, name: str):
    if timing is None:
        yield
        return
    with timing.phase(name):
        yield


__all__ = ["TimingRecorder", "timed_phase"]
