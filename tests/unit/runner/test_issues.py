from __future__ import annotations

import json

import pytest

from automl.runner.issues import IssueRecorder

pytestmark = pytest.mark.unit


def test_record_exception_captures_class_message_and_phase():
    recorder = IssueRecorder()
    try:
        raise ValueError("boom")
    except ValueError as exc:
        recorder.record(exc, phase="evaluation", severity="warning")
    (issue,) = recorder.snapshot()
    assert issue["phase"] == "evaluation"
    assert issue["severity"] == "warning"
    assert issue["error_class"] == "ValueError"
    assert issue["message"] == "boom"
    assert issue["traceback_tail"]
    assert recorder.count == 1


def test_record_plain_message():
    recorder = IssueRecorder()
    recorder.record("latency not measured", phase="validation_publish")
    (issue,) = recorder.snapshot()
    assert issue["severity"] == "error"
    assert issue["error_class"] == ""
    assert issue["message"] == "latency not measured"


def test_jsonl_appended_as_events_happen(tmp_path):
    jsonl = tmp_path / "issues.jsonl"
    recorder = IssueRecorder(jsonl_path=jsonl)
    recorder.record("first", phase="fit")
    recorder.record("second", phase="evaluation")
    lines = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    assert [line["message"] for line in lines] == ["first", "second"]


def test_recording_never_raises_when_jsonl_unwritable(tmp_path):
    recorder = IssueRecorder(jsonl_path=tmp_path / "no-such-dir" / "issues.jsonl")
    recorder.record("still recorded in memory", phase="fit")
    assert recorder.count == 1


def test_snapshot_is_json_serializable():
    recorder = IssueRecorder()
    try:
        raise RuntimeError("x")
    except RuntimeError as exc:
        recorder.record(exc, phase="fit")
    json.dumps(recorder.snapshot())


def test_record_survives_exception_whose_str_raises():
    class _EvilError(Exception):
        def __str__(self):
            raise RuntimeError("nope")

    recorder = IssueRecorder()
    recorder.record(_EvilError(), phase="fit")
    (issue,) = recorder.snapshot()
    assert issue["error_class"] == "_EvilError"
    assert "unprintable" in issue["message"]


def test_try_log_train_eval_records_issue_instead_of_swallowing(monkeypatch):
    from automl.runner import trial as trial_module
    from automl.runner.context import TrialContext

    def explode(**kwargs):
        raise RuntimeError("train eval blew up")

    monkeypatch.setattr(trial_module, "prepare_eval_dataset", explode)
    ctx = TrialContext()
    trial_module._try_log_train_eval(
        ctx=ctx,
        run_id="run1",
        active=object(),
        model=object(),
        dataset_id="ds_001",
        train_split="train",
        feature_registry=object(),
    )
    (issue,) = ctx.issues.snapshot()
    assert issue["severity"] == "warning"
    assert issue["error_class"] == "RuntimeError"
    assert "train eval blew up" in issue["message"]
