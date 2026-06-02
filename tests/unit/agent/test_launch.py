import json
from pathlib import Path

import pytest

from automl.eval import Auc, EvalSpec
from automl.project import (
    BinaryClassification,
    ModelRoute,
    ModelsConfig,
    ProjectConfig,
    RunConfig,
    Session,
    Splits,
)

pytestmark = pytest.mark.unit


def _session(tmp_path: Path) -> Session:
    _write_agent(
        tmp_path / "agent-skills" / "agents" / "automl-proposer.md",
        name="automl-proposer",
        prompt="Propose one trial.",
    )
    _write_agent(
        tmp_path / "agent-skills" / "agents" / "automl-coder.md",
        name="automl-coder",
        prompt="Code one trial.",
    )
    return Session(
        config=ProjectConfig(
            project_name="demo",
            repo_root=tmp_path,
            project_dir=tmp_path / "projects" / "demo",
            config_path=tmp_path / "projects" / "demo" / "config.py",
            task=BinaryClassification(target="target"),
            eval_spec=EvalSpec(primary=Auc()),
            run_config=RunConfig(
                experiment_id="exp",
                splits=Splits({"train": ((0, 80),), "test": ((80, 100),)}),
                models=ModelsConfig(
                    manager=ModelRoute("manager-model", "low"),
                    proposer=ModelRoute("proposer-model", "medium"),
                    coder=ModelRoute("coder-model", "high"),
                ),
                per_trial_seconds=120,
            ),
        ),
        dry_run=True,
    )


def _write_agent(path: Path, *, name: str, prompt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""---
name: {name}
description: fixture
tools: Read, Bash
model: inherit
effort: high
---

{prompt}
""",
        encoding="utf-8",
    )


def test_build_launch_uses_session_model_routes_and_agent_overrides(monkeypatch, tmp_path):
    from automl.agent.launch import build_launch

    monkeypatch.setenv("AUTOML_SESSION_ID", "session-123")
    active = _session(tmp_path)

    launch = build_launch(
        session=active,
        automl_args=[],
        max_budget_usd="1",
        output_format="stream-json",
        claude_bin="claude-test",
    )

    assert launch.cwd == tmp_path
    assert launch.env["AUTOML_SESSION_ID"] == "session-123"
    assert launch.env["CLAUDE_SESSION_ID"] == "session-123"
    assert launch.env["AUTOML_PROJECT_ROOT"] == str(tmp_path)
    assert launch.env["AUTOML_PROJECT"] == "demo"
    assert launch.env["AUTOML_EXPERIMENT_ID"] == "exp"
    assert launch.env["AUTOML_INHERIT_DRY_RUN"] == "1"
    assert launch.command[:6] == [
        "claude-test",
        "--session-id",
        "session-123",
        "--model",
        "manager-model",
        "--effort",
    ]
    assert "low" in launch.command
    assert "--verbose" in launch.command
    assert launch.command[-1] == "/brigit-automl:automl experiment run --project demo"

    agents = json.loads(launch.command[launch.command.index("--agents") + 1])
    assert agents["automl-proposer"]["model"] == "proposer-model"
    assert agents["automl-proposer"]["effort"] == "medium"
    assert agents["automl-proposer"]["tools"] == ["Read", "Bash"]
    assert agents["automl-proposer"]["prompt"] == "Propose one trial."
    assert agents["automl-coder"]["model"] == "coder-model"
    assert agents["automl-coder"]["effort"] == "high"


def test_build_launch_rejects_mismatched_inner_project(tmp_path):
    from automl.agent.launch import build_launch

    with pytest.raises(ValueError, match="does not match"):
        build_launch(
            session=_session(tmp_path),
            automl_args=["experiment", "run", "--project", "other"],
            max_budget_usd="1",
            output_format="json",
        )


def test_launch_command_preserves_explicit_loop_options(monkeypatch, tmp_path):
    from automl.agent.launch import build_launch

    monkeypatch.setenv("AUTOML_SESSION_ID", "session-123")

    launch = build_launch(
        session=_session(tmp_path),
        automl_args=[
            "experiment",
            "run",
            "--project",
            "demo",
            "--max-iter",
            "7",
            "--time-budget",
            "1.5",
            "--instruction",
            "prefer linear models",
        ],
        max_budget_usd="1",
        output_format="json",
    )

    expected_prompt = (
        "/brigit-automl:automl experiment run --project demo --max-iter 7 "
        "--time-budget 1.5 --instruction 'prefer linear models'"
    )
    assert launch.command[-1] == expected_prompt
