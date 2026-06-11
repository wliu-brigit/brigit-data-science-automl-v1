"""Regression tests: log_issue_artifacts is called on both _run_trial exit paths.

Both the success path (line ~238 after log_manifest) and the failure path
(inside _publish_failure_artifacts ~line 305) must call log_issue_artifacts.
Deleting either call site must fail one of these tests.
"""

from __future__ import annotations

import pytest
import pandas as pd

import automl.runner.trial as trial_module
from automl.runner.trial import TrialExecutionContext

pytestmark = pytest.mark.unit


class _FakeRunConfig:
    train_split = "train"
    eval_split = "test"

    class splits:
        class predicates:
            @staticmethod
            def items():
                return []


class _FakeEvalSpec:
    def validate_columns(self, df, target_col):
        pass


class _FakeConfig:
    target_column = "target"
    project_package = "projects.demo"

    def require_run_config(self):
        return _FakeRunConfig()

    def require_eval_spec(self):
        return _FakeEvalSpec()


class _FakeSession:
    config = _FakeConfig()
    project_name = "demo"
    active_experiment_id = "exp1"
    dry_run = True
    namespace = "qa"


class _FakeLoaded:
    df = pd.DataFrame({"target": [0, 1] * 10, "feat": range(20)})
    registry = object()
    id = "ds_001"

    @property
    def dataset(self):
        return object()


class _FakeModelRef:
    logged_uri = "runs:/run_abc/model"


class _FakeEvalResult:
    label = "test"
    predictions_uri = "gs://bucket/preds"

    def to_dict(self):
        return {}


class _FakeEvalDataset:
    id = "eval_ds_001"
    df = pd.DataFrame()


class _FakeModel:
    name = "fake"
    feature_registry = None

    def fit(self, df, registry, seed=0):
        return None


class _FakeModelCls:
    def __call__(self):
        return _FakeModel()


class _FakeRun:
    """Context manager that yields a fixed run_id without touching MLflow."""

    def __enter__(self):
        return "run_abc"

    def __exit__(self, *a):
        return False


def _patch_success_path(monkeypatch):
    """Patch _run_trial so the success path completes normally."""
    monkeypatch.setattr(trial_module.data, "load_dataset", lambda **kw: _FakeLoaded())
    monkeypatch.setattr(trial_module, "_load_model_class", lambda ctx: _FakeModelCls())
    monkeypatch.setattr(trial_module, "validate_model", lambda *a, **kw: object())
    monkeypatch.setattr(trial_module, "require_validation_passed", lambda r: None)
    monkeypatch.setattr(trial_module.mlflow_experiment, "ensure", lambda **kw: None)
    monkeypatch.setattr(trial_module.mlflow_experiment, "next_trial_number", lambda **kw: 1)
    monkeypatch.setattr(trial_module.mlflow_trial, "active", lambda **kw: _FakeRun())
    monkeypatch.setattr(trial_module.mlflow_trial, "set_tags", lambda *a, **kw: None)
    monkeypatch.setattr(trial_module.mlflow_trial, "log_param", lambda *a, **kw: None)
    monkeypatch.setattr(trial_module, "validate_fitted_model", lambda *a, **kw: None)
    monkeypatch.setattr(trial_module, "log_agent_proposal", lambda **kw: False)
    monkeypatch.setattr(trial_module, "log_feature_artifacts", lambda **kw: None)
    monkeypatch.setattr(trial_module, "log_model", lambda **kw: _FakeModelRef())
    monkeypatch.setattr(
        trial_module, "prepare_eval_dataset", lambda **kw: (_FakeEvalDataset(), None)
    )
    monkeypatch.setattr(trial_module, "evaluate", lambda **kw: _FakeEvalResult())
    monkeypatch.setattr(trial_module, "_try_log_train_eval", lambda **kw: None)
    monkeypatch.setattr(trial_module, "_trial_data_contract", lambda **kw: object())
    monkeypatch.setattr(trial_module, "log_data_contract", lambda *a, **kw: None)
    monkeypatch.setattr(
        trial_module, "log_validation_artifacts", lambda **kw: {"status": "passed"}
    )
    monkeypatch.setattr(
        trial_module, "build_runner_timing_summary", lambda s: {"schema_version": 2}
    )
    monkeypatch.setattr(trial_module, "log_timing", lambda *a, **kw: None)
    monkeypatch.setattr(trial_module, "log_manifest", lambda **kw: None)
    monkeypatch.setattr(trial_module, "scalar_metric_records", lambda d: {})


def test_log_issue_artifacts_called_on_success_path(monkeypatch):
    """Deleting the log_issue_artifacts call on the success path must fail here."""
    issue_calls: list[str] = []

    monkeypatch.setattr(
        trial_module,
        "log_issue_artifacts",
        lambda run_id, issues: issue_calls.append(run_id),
    )
    _patch_success_path(monkeypatch)

    ctx = TrialExecutionContext(session=_FakeSession())
    result = trial_module._run_trial(ctx)

    assert result.status == "FINISHED"
    assert issue_calls == ["run_abc"], (
        "log_issue_artifacts must be called on the success path; "
        f"calls observed: {issue_calls}"
    )


def test_log_issue_artifacts_called_on_failure_path(monkeypatch):
    """Deleting the log_issue_artifacts call inside _publish_failure_artifacts must fail here."""
    issue_calls: list[str] = []

    monkeypatch.setattr(
        trial_module,
        "log_issue_artifacts",
        lambda run_id, issues: issue_calls.append(run_id),
    )
    # Patch enough to reach the active run context manager before failing.
    monkeypatch.setattr(trial_module.data, "load_dataset", lambda **kw: _FakeLoaded())
    monkeypatch.setattr(trial_module, "_load_model_class", lambda ctx: _FakeModelCls())
    monkeypatch.setattr(trial_module, "validate_model", lambda *a, **kw: object())
    monkeypatch.setattr(trial_module, "require_validation_passed", lambda r: None)
    monkeypatch.setattr(trial_module.mlflow_experiment, "ensure", lambda **kw: None)
    monkeypatch.setattr(trial_module.mlflow_experiment, "next_trial_number", lambda **kw: 2)
    monkeypatch.setattr(trial_module.mlflow_trial, "active", lambda **kw: _FakeRun())
    monkeypatch.setattr(trial_module.mlflow_trial, "set_tags", lambda *a, **kw: None)
    monkeypatch.setattr(trial_module.mlflow_trial, "log_param", lambda *a, **kw: None)
    monkeypatch.setattr(trial_module, "log_agent_proposal", lambda **kw: False)
    monkeypatch.setattr(trial_module, "log_failure_artifacts", lambda **kw: None)
    # Blow up during validate_fitted_model — we have run_id at this point.
    monkeypatch.setattr(
        trial_module, "validate_fitted_model", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("forced fit failure"))
    )

    ctx = TrialExecutionContext(session=_FakeSession())
    result = trial_module._run_trial(ctx)

    assert result.status == "FAILED"
    assert "run_abc" in issue_calls, (
        "log_issue_artifacts must be called on the failure path; "
        f"calls observed: {issue_calls}"
    )


def test_enable_faulthandler_survives_stderr_without_fileno(monkeypatch):
    """In-process CLI calls under sys-level capture have a fileno-less stderr;
    faulthandler is best-effort diagnostics and must never fail the trial."""
    import io
    import sys

    monkeypatch.setattr(sys, "stderr", io.StringIO())
    trial_module._enable_faulthandler()
