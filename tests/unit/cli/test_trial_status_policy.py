import argparse
from pathlib import Path

import pytest

from automl.cli import _trial_actions as trial_cli
from automl.trial import TrialStatus

pytestmark = pytest.mark.unit


def test_runner_result_exit_code_owns_finished_policy():
    from automl.runner.results import trial_result_exit_code

    assert trial_result_exit_code({"status": "success"}) == 1
    assert trial_result_exit_code({"status": TrialStatus.FINISHED.value}) == 0


def test_trial_cli_uses_runner_exit_policy(monkeypatch):
    active = object()
    result = {"status": "CUSTOM"}
    calls = []

    monkeypatch.setattr(trial_cli, "session_from_args", lambda args: active)
    monkeypatch.setattr(trial_cli, "run_trial", lambda path, **kwargs: result)
    monkeypatch.setattr(
        trial_cli,
        "trial_result_exit_code",
        lambda value: calls.append(value) or 7,
        raising=False,
    )
    monkeypatch.setattr(trial_cli, "print_json", lambda value: None)

    assert trial_cli._run(argparse.Namespace(path="trial-one")) == 7
    assert calls == [result]


def test_trial_cli_exit_policy_uses_canonical_finished_status(monkeypatch, tmp_path):
    active = object()
    model_path = tmp_path / "model.py"
    model_path.write_text("class Model:\n    pass\n")
    monkeypatch.setattr(trial_cli, "session_from_args", lambda args: active)
    monkeypatch.setattr(trial_cli, "print_json", lambda value: None)

    monkeypatch.setattr(trial_cli, "run_trial", lambda path, **kwargs: {"status": "success"})
    assert trial_cli._run(argparse.Namespace(path="trial-one")) == 1

    monkeypatch.setattr(
        trial_cli,
        "run_trial",
        lambda path, **kwargs: {"status": TrialStatus.FINISHED.value},
    )
    assert trial_cli._run(argparse.Namespace(path="trial-one")) == 0

    promote_args = argparse.Namespace(
        slug="trial-one",
        model_path=model_path,
        hypothesis="manual",
        strategy="manual_promote",
    )
    monkeypatch.setattr(trial_cli, "create_trial", lambda **kwargs: Path("trial-one"))
    monkeypatch.setattr(trial_cli, "run_trial", lambda path, **kwargs: {"status": "success"})
    assert trial_cli._promote(promote_args) == 1

    monkeypatch.setattr(
        trial_cli,
        "run_trial",
        lambda path, **kwargs: {"status": TrialStatus.FINISHED.value},
    )
    assert trial_cli._promote(promote_args) == 0


def test_trial_cli_promote_composes_create_and_run(monkeypatch, tmp_path):
    active = object()
    model_path = tmp_path / "model.py"
    model_path.write_text("class Model:\n    pass\n")
    trial_path = tmp_path / "trial-one"
    calls = []

    monkeypatch.setattr(trial_cli, "session_from_args", lambda args: active)
    monkeypatch.setattr(trial_cli, "print_json", lambda value: calls.append(("print", value)))

    def fake_create_trial(**kwargs):
        calls.append(("create", kwargs))
        return trial_path

    def fake_run_trial(path, **kwargs):
        calls.append(("run", {"path": path, **kwargs}))
        return {"status": TrialStatus.FINISHED.value, "trial_dir": str(path)}

    monkeypatch.setattr(trial_cli, "create_trial", fake_create_trial)
    monkeypatch.setattr(trial_cli, "run_trial", fake_run_trial)

    exit_code = trial_cli._promote(
        argparse.Namespace(
            slug="manual_promote",
            model_path=model_path,
            hypothesis="Promote a reviewed model file.",
            strategy="manual_promote",
        )
    )

    assert exit_code == 0
    assert calls == [
        (
            "create",
            {
                "slug": "manual_promote",
                "strategy": "manual_promote",
                "hypothesis": "Promote a reviewed model file.",
                "model_source": model_path,
                "training_origin": "human",
                "session": active,
            },
        ),
        ("run", {"path": trial_path, "session": active}),
        ("print", {"status": TrialStatus.FINISHED.value, "trial_dir": str(trial_path)}),
    ]
