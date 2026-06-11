from __future__ import annotations

import json
from pathlib import Path

import pytest

from automl.mlflow import tags as mlflow_tags
from automl.runner.context import TrialContext
from automl.runner import issue_artifacts

pytestmark = pytest.mark.unit


def test_publishes_issues_json_and_count_tag(monkeypatch):
    published = {}

    def fake_write_local_file(run_id, artifact_path, local_path):
        published["run_id"] = run_id
        published["artifact_path"] = artifact_path
        published["payload"] = json.loads(Path(local_path).read_text(encoding="utf-8"))

    tags_set = {}
    monkeypatch.setattr(
        issue_artifacts.runner_artifacts, "write_local_file", fake_write_local_file
    )
    monkeypatch.setattr(
        issue_artifacts.mlflow_trial,
        "set_tags",
        lambda run_id, tags: tags_set.update(tags),
    )

    ctx = TrialContext()
    ctx.record_issue("something best-effort failed", phase="evaluation", severity="warning")
    issue_artifacts.log_issue_artifacts("run123", ctx.issues)

    assert published["run_id"] == "run123"
    assert published["artifact_path"] == "trial/issues.json"
    assert published["payload"]["schema_version"] == 1
    assert published["payload"]["issues"][0]["message"] == "something best-effort failed"
    assert tags_set[mlflow_tags.TRIAL_ISSUE_COUNT] == 1


def test_no_run_id_is_a_noop(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("must not publish without a run")

    monkeypatch.setattr(issue_artifacts.runner_artifacts, "write_local_file", explode)
    ctx = TrialContext()
    ctx.record_issue("x", phase="fit")
    issue_artifacts.log_issue_artifacts("", ctx.issues)


def test_zero_issues_still_publishes_count_zero(monkeypatch):
    published = {}
    tags_set = {}
    monkeypatch.setattr(
        issue_artifacts.runner_artifacts,
        "write_local_file",
        lambda run_id, artifact_path, local_path: published.update(
            payload=json.loads(Path(local_path).read_text(encoding="utf-8"))
        ),
    )
    monkeypatch.setattr(
        issue_artifacts.mlflow_trial,
        "set_tags",
        lambda run_id, tags: tags_set.update(tags),
    )
    issue_artifacts.log_issue_artifacts("run123", TrialContext().issues)
    assert published["payload"]["issues"] == []
    assert tags_set[mlflow_tags.TRIAL_ISSUE_COUNT] == 0
