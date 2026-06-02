from __future__ import annotations

import pytest

from automl.runner.timing import TimingRecorder, timed_phase

pytestmark = pytest.mark.unit


def test_snapshot_shape(monkeypatch):
    ticks = iter([10.0, 12.5])
    monkeypatch.setattr("automl.runner.timing.time.monotonic", lambda: next(ticks))

    snapshot = TimingRecorder().snapshot()

    assert snapshot["schema_version"] == 1
    assert snapshot["unit"] == "seconds"
    assert snapshot["phases"] == {}
    assert snapshot["total_seconds"] == 2.5


def test_repeated_phase_names_accumulate_duration(monkeypatch):
    ticks = iter([0.0, 1.0, 2.0, 3.0, 4.0, 4.0])
    monkeypatch.setattr("automl.runner.timing.time.monotonic", lambda: next(ticks))
    timing = TimingRecorder()

    with timing.phase("fit"):
        pass
    with timed_phase(timing, "fit"):
        pass

    assert timing.last_phase == "fit"
    assert timing.snapshot()["phases"] == {"fit": 2.0}


def test_timed_phase_none_is_noop_context_manager():
    with timed_phase(None, "fit"):
        value = "entered"

    assert value == "entered"
