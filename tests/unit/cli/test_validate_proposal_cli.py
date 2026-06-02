import json
import types

import pytest

pytestmark = pytest.mark.unit


def test_validate_proposal_cli_writes_report_and_validated_json(monkeypatch, tmp_path, capsys):
    from automl.cli import main

    monkeypatch.setattr(
        "automl.agent.checks.allowed_dependencies",
        lambda session=None: ["pandas"],
    )
    proposal_path = tmp_path / "proposal.json"
    output_path = tmp_path / "validated.json"
    proposal_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "slug": "baseline",
                "strategy": "baseline",
                "hypothesis": "Try a baseline.",
                "implementation_plan": ["Train a simple model."],
                "constraints": ["Do not read test data."],
                "required_dependencies": ["pandas"],
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "validate",
                "proposal",
                "--proposal-json",
                str(proposal_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is True
    assert json.loads(output_path.read_text(encoding="utf-8"))["slug"] == "baseline"


def test_validate_proposal_cli_uses_inferred_session_when_available(
    monkeypatch,
    tmp_path,
    capsys,
):
    from automl.cli import main
    from automl.validate import ValidationReport

    active = object()
    calls = []
    monkeypatch.setattr("automl.cli._validate_actions.session_from_args", lambda args: active)

    def fake_validate_proposal(**kwargs):
        calls.append(kwargs)
        return ValidationReport()

    monkeypatch.setattr("automl.cli._validate_actions.validate_proposal", fake_validate_proposal)
    proposal_path = tmp_path / "proposal.json"
    proposal_path.write_text("{}", encoding="utf-8")

    assert (
        main(
            [
                "--project-root",
                str(tmp_path),
                "validate",
                "proposal",
                "--proposal-json",
                str(proposal_path),
            ]
        )
        == 0
    )

    assert calls[0]["session"] is active
    assert json.loads(capsys.readouterr().out)["passed"] is True


def test_validate_proposal_cli_propagates_explicit_session_errors(monkeypatch, tmp_path):
    from automl.cli import main
    from automl.errors import ProjectError

    monkeypatch.setattr(
        "automl.cli._validate_actions.session_from_args",
        lambda args: (_ for _ in ()).throw(ProjectError("project missing")),
    )
    proposal_path = tmp_path / "proposal.json"
    proposal_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ProjectError, match="project missing"):
        main(
            [
                "--project",
                "missing",
                "validate",
                "proposal",
                "--proposal-json",
                str(proposal_path),
            ]
        )


def test_experiment_proposer_context_cli_uses_session_and_prints_json(
    monkeypatch,
    tmp_path,
    capsys,
):
    from automl.cli import main

    active = object()
    monkeypatch.setattr("automl.cli._common.use_project", lambda *args, **kwargs: active)
    monkeypatch.setattr(
        "automl.cli._experiment_actions.gather_proposer_context",
        lambda **kwargs: {
            "project_name": "demo",
            "metric": kwargs["metric"],
            "session_is_active": kwargs["session"] is active,
        },
    )

    assert (
        main(
            [
                "--project",
                "demo",
                "--project-root",
                str(tmp_path),
                "experiment",
                "proposer-context",
                "--metric",
                "auc",
            ]
        )
        == 0
    )

    assert json.loads(capsys.readouterr().out) == {
        "project_name": "demo",
        "metric": "auc",
        "session_is_active": True,
    }


def test_experiment_run_cli_launches_subprocess(monkeypatch, tmp_path):
    from automl.agent.launch import LaunchSpec
    from automl.cli import main

    active = types.SimpleNamespace(project_name="demo")
    captured = {}
    monkeypatch.setattr("automl.cli._common.use_project", lambda *args, **kwargs: active)
    monkeypatch.setattr(
        "automl.cli._experiment_actions.build_launch",
        lambda **kwargs: LaunchSpec(
            command=["claude-test", "payload"],
            env={"AUTOML_SESSION_ID": "session-1"},
            cwd=tmp_path,
        ),
    )

    def fake_run(command, *, env, cwd, check):
        captured.update({"command": command, "env": env, "cwd": cwd, "check": check})
        return types.SimpleNamespace(returncode=7)

    monkeypatch.setattr("automl.cli._experiment_actions.subprocess.run", fake_run)

    assert (
        main(
            [
                "--project",
                "demo",
                "--project-root",
                str(tmp_path),
                "experiment",
                "run",
                "--max-budget-usd",
                "1",
                "--output-format",
                "json",
            ]
        )
        == 7
    )
    assert captured == {
        "command": ["claude-test", "payload"],
        "env": {"AUTOML_SESSION_ID": "session-1"},
        "cwd": tmp_path,
        "check": False,
    }
