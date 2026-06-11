from __future__ import annotations

import pytest

from automl.runner.context import TrialContext

pytestmark = pytest.mark.unit


def test_phase_delegates_to_timing():
    ctx = TrialContext()
    with ctx.phase("fit"):
        pass
    assert "fit" in ctx.timing.phases
    assert ctx.timing.last_phase == "fit"


def test_record_issue_defaults_phase_to_last_timing_phase():
    ctx = TrialContext()
    with ctx.phase("evaluation"):
        pass
    ctx.record_issue("went sideways", severity="warning")
    (issue,) = ctx.issues.snapshot()
    assert issue["phase"] == "evaluation"
    assert issue["severity"] == "warning"


def test_record_issue_with_explicit_phase():
    ctx = TrialContext()
    ctx.record_issue("early problem", phase="model_import")
    (issue,) = ctx.issues.snapshot()
    assert issue["phase"] == "model_import"


def test_jsonl_lands_in_trial_dir(tmp_path):
    ctx = TrialContext(trial_dir=tmp_path)
    ctx.record_issue("evidence", phase="fit")
    assert (tmp_path / "issues.jsonl").exists()


def test_identity_fields_fill_in_as_known():
    ctx = TrialContext()
    ctx.run_id = "run1"
    ctx.trial_id = "1_slug"
    ctx.trial_number = 1
    ctx.slug = "slug"
    ctx.strategy = "baseline"
    assert ctx.run_id == "run1"
