from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import types

import pytest

pytestmark = pytest.mark.unit


def _patch_attr(monkeypatch, module_name: str, name: str, value) -> None:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        return
    monkeypatch.setattr(module, name, value, raising=False)


def _patch_use_project(monkeypatch, active):
    calls = []

    def fake_use_project(name, **kwargs):
        calls.append({"name": name, **kwargs})
        return active

    _patch_attr(monkeypatch, "automl.cli", "use_project", fake_use_project)
    _patch_attr(monkeypatch, "automl.cli._common", "use_project", fake_use_project)
    return calls


def _subparser(parser: argparse.ArgumentParser, *path: str) -> argparse.ArgumentParser:
    current = parser
    for name in path:
        subparser_action = next(
            action for action in current._actions if isinstance(action, argparse._SubParsersAction)
        )
        current = subparser_action.choices[name]
    return current


def _option_strings(parser: argparse.ArgumentParser) -> set[str]:
    return {option for action in parser._actions for option in action.option_strings}


def test_json_flag_is_reserved_for_experiment_run_output():
    from automl.cli import build_parser

    parser = build_parser()

    assert "--json" in _option_strings(_subparser(parser, "experiment", "run"))

    jsonless_commands = [
        ("project", "list"),
        ("project", "deps"),
        ("project", "init"),
        ("project", "delete"),
        ("experiment", "list"),
        ("experiment", "delete"),
        ("experiment", "leaderboard"),
        ("experiment", "compare"),
        ("experiment", "summary"),
        ("experiment", "proposer-context"),
        ("trial", "list"),
        ("trial", "create"),
        ("trial", "fork"),
        ("trial", "promote"),
        ("trial", "run"),
        ("trial", "show"),
        ("trial", "delete"),
        ("trial", "lock", "acquire"),
        ("trial", "lock", "release"),
        ("data", "list"),
        ("data", "profile"),
        ("data", "materialize"),
        ("eval", "list"),
        ("eval", "compute"),
        ("validate", "project"),
        ("validate", "model"),
        ("validate", "proposal"),
    ]
    for command in jsonless_commands:
        assert "--json" not in _option_strings(_subparser(parser, *command)), command


def test_validate_proposal_uses_named_input_path_not_output_json_flag():
    from automl.cli import build_parser

    proposal_parser = _subparser(build_parser(), "validate", "proposal")

    assert "--proposal-json" in _option_strings(proposal_parser)
    assert "--json" not in _option_strings(proposal_parser)


def test_root_session_flags_apply_to_experiment_proposer_context(monkeypatch, tmp_path, capsys):
    from automl.cli import main

    active = types.SimpleNamespace(
        active_experiment_id="exp",
        config=types.SimpleNamespace(repo_root=tmp_path),
    )
    calls = _patch_use_project(monkeypatch, active)
    _patch_attr(
        monkeypatch,
        "automl.cli._experiment_actions",
        "gather_proposer_context",
        lambda **kwargs: {
            "session_is_active": kwargs["session"] is active,
            "metric": kwargs["metric"],
        },
    )

    assert (
        main(
            [
                "--project",
                "demo",
                "--project-root",
                str(tmp_path),
                "--dry-run",
                "--namespace",
                "qa",
                "--experiment-id",
                "exp-1",
                "experiment",
                "proposer-context",
                "--metric",
                "test.auc",
            ]
        )
        == 0
    )

    assert calls == [
        {
            "name": "demo",
            "repo_root": tmp_path,
            "dry_run": True,
            "namespace": "qa",
            "experiment_id": "exp-1",
        }
    ]
    assert json.loads(capsys.readouterr().out) == {
        "session_is_active": True,
        "metric": "test.auc",
    }


def test_project_catalog_verbs_dispatch_to_library(monkeypatch, tmp_path, capsys):
    from automl.cli import main

    active = types.SimpleNamespace(
        active_experiment_id="exp",
        config=types.SimpleNamespace(repo_root=tmp_path),
    )
    _patch_use_project(monkeypatch, active)
    _patch_attr(monkeypatch, "automl.cli.project", "list_projects", lambda **kwargs: ["demo"])
    _patch_attr(
        monkeypatch, "automl.cli.project", "allowed_dependencies", lambda **kwargs: ["pandas"]
    )
    _patch_attr(
        monkeypatch,
        "automl.cli.project",
        "create_project",
        lambda *args, **kwargs: {"project": args[0], "created": ["projects/new/config.py"]},
    )
    _patch_attr(
        monkeypatch,
        "automl.cli.project",
        "delete_project",
        lambda name, **kwargs: {
            "scope": "project",
            "name": name,
            "session": kwargs["session"] is active,
            "backend_store_uri": kwargs["backend_store_uri"],
            "artifacts_destination": kwargs["artifacts_destination"],
        },
    )

    assert main(["--project-root", str(tmp_path), "project", "list"]) == 0
    assert json.loads(capsys.readouterr().out) == ["demo"]

    assert main(["--project", "demo", "--project-root", str(tmp_path), "project", "deps"]) == 0
    assert json.loads(capsys.readouterr().out) == ["pandas"]

    assert main(["--project-root", str(tmp_path), "project", "init", "new"]) == 0
    assert json.loads(capsys.readouterr().out)["project"] == "new"

    assert (
        main(
            [
                "--project",
                "demo",
                "--project-root",
                str(tmp_path),
                "project",
                "delete",
                "demo",
                "--apply",
                "--backend-store-uri",
                "sqlite:////tmp/mlflow.db",
                "--artifacts-destination",
                "gs://bucket/mlflow-artifacts",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {
        "scope": "project",
        "name": "demo",
        "session": True,
        "backend_store_uri": "sqlite:////tmp/mlflow.db",
        "artifacts_destination": "gs://bucket/mlflow-artifacts",
    }


def test_cleanup_hard_delete_options_forward_to_experiment_and_trial(monkeypatch, tmp_path, capsys):
    from automl.cli import main

    active = types.SimpleNamespace(
        active_experiment_id="exp",
        config=types.SimpleNamespace(repo_root=tmp_path),
    )
    _patch_use_project(monkeypatch, active)
    experiment_calls = []
    trial_calls = []

    def fake_delete_experiment(experiment_id, **kwargs):
        experiment_calls.append({"experiment_id": experiment_id, **kwargs})
        return {"deleted": experiment_id}

    def fake_delete_trial(run_id, **kwargs):
        trial_calls.append({"run_id": run_id, **kwargs})
        return {"deleted": run_id}

    _patch_attr(
        monkeypatch,
        "automl.cli._experiment_actions",
        "delete_experiment",
        fake_delete_experiment,
    )
    _patch_attr(monkeypatch, "automl.cli._trial_actions", "delete_trial", fake_delete_trial)

    assert (
        main(
            [
                "--project",
                "demo",
                "--project-root",
                str(tmp_path),
                "experiment",
                "delete",
                "exp",
                "--apply",
                "--hard-delete",
                "--backend-store-uri",
                "sqlite:////tmp/mlflow.db",
                "--artifacts-destination",
                "gs://bucket/mlflow-artifacts",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {"deleted": "exp"}

    assert (
        main(
            [
                "--project",
                "demo",
                "--project-root",
                str(tmp_path),
                "trial",
                "delete",
                "run-1",
                "--apply",
                "--hard-delete",
                "--backend-store-uri",
                "sqlite:////tmp/mlflow.db",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {"deleted": "run-1"}

    assert experiment_calls == [
        {
            "experiment_id": "exp",
            "apply": True,
            "hard_delete": True,
            "backend_store_uri": "sqlite:////tmp/mlflow.db",
            "artifacts_destination": "gs://bucket/mlflow-artifacts",
            "session": active,
        }
    ]
    assert trial_calls == [
        {
            "run_id": "run-1",
            "apply": True,
            "hard_delete": True,
            "backend_store_uri": "sqlite:////tmp/mlflow.db",
            "artifacts_destination": "",
            "session": active,
        }
    ]


def test_experiment_trial_data_eval_validate_catalog_verbs(monkeypatch, tmp_path, capsys):
    from automl.agent.launch import LaunchSpec
    from automl.cli import main
    from automl.validate import ValidationReport

    active = types.SimpleNamespace(
        active_experiment_id="exp",
        config=types.SimpleNamespace(repo_root=tmp_path),
        project_name="demo",
    )
    _patch_use_project(monkeypatch, active)
    run_calls = []
    validate_model_calls = []

    _patch_attr(
        monkeypatch,
        "automl.cli._experiment_actions",
        "list_experiments",
        lambda **kwargs: [{"experiment_id": "exp"}],
    )
    _patch_attr(
        monkeypatch,
        "automl.cli._experiment_actions",
        "leaderboard",
        lambda **kwargs: {"kind": "leaderboard", "session": kwargs["session"] is active},
    )
    _patch_attr(
        monkeypatch,
        "automl.cli._experiment_actions",
        "compare",
        lambda run_ids, **kwargs: {
            "run_ids": list(run_ids),
            "session": kwargs["session"] is active,
        },
    )
    _patch_attr(
        monkeypatch,
        "automl.cli._experiment_actions",
        "build_summary",
        lambda **kwargs: {
            "summary_kind": "experiment_summary",
            "session": kwargs["session"] is active,
        },
    )
    _patch_attr(
        monkeypatch,
        "automl.cli._experiment_actions",
        "delete_experiment",
        lambda experiment_id, **kwargs: {
            "deleted": experiment_id,
            "session": kwargs["session"] is active,
        },
    )
    _patch_attr(
        monkeypatch,
        "automl.cli._experiment_actions",
        "build_launch",
        lambda **kwargs: LaunchSpec(
            command=["claude-test"], env={"AUTOML_INHERIT_DRY_RUN": "1"}, cwd=tmp_path
        ),
    )
    monkeypatch.setattr(
        "automl.cli._experiment_actions.subprocess.run",
        lambda command, **kwargs: (
            run_calls.append((command, kwargs)) or subprocess.CompletedProcess(command, 0)
        ),
    )
    _patch_attr(
        monkeypatch,
        "automl.cli._experiment_actions",
        "gather_proposer_context",
        lambda **kwargs: {"context": True, "session": kwargs["session"] is active},
    )

    _patch_attr(
        monkeypatch,
        "automl.cli._trial_actions",
        "list_trials",
        lambda **kwargs: [{"run_id": "run-1"}],
    )
    _patch_attr(
        monkeypatch,
        "automl.cli._trial_actions",
        "create_trial",
        lambda **kwargs: tmp_path / "trial-one",
    )
    _patch_attr(
        monkeypatch,
        "automl.cli._trial_actions",
        "fork_trial",
        lambda **kwargs: tmp_path / "trial-two",
    )
    _patch_attr(
        monkeypatch,
        "automl.cli._trial_actions",
        "run_trial",
        lambda path, **kwargs: {
            "path": str(path),
            "session": kwargs["session"] is active,
            "status": "FINISHED",
        },
    )
    _patch_attr(
        monkeypatch,
        "automl.cli._trial_actions",
        "show_trial",
        lambda run_id, **kwargs: {"run_id": run_id, "session": kwargs["session"] is active},
    )
    _patch_attr(
        monkeypatch,
        "automl.cli._trial_actions",
        "delete_trial",
        lambda run_id, **kwargs: {"deleted": run_id, "session": kwargs["session"] is active},
    )
    _patch_attr(
        monkeypatch,
        "automl.cli._trial_actions",
        "trial_lock",
        types.SimpleNamespace(
            acquire_for_session=lambda active, session_id: {
                "status": "acquired",
                "session_id": session_id,
                "route": "demo/exp",
                "lock_id": "lock-1",
            },
            release_for_session=lambda active, session_id, lock_id: {
                "status": "released",
                "session_id": session_id,
                "lock_id": lock_id,
            },
        ),
    )

    _patch_attr(monkeypatch, "automl.cli.data", "list_datasets", lambda **kwargs: {"datasets": []})
    _patch_attr(
        monkeypatch,
        "automl.cli.data",
        "profile",
        lambda **kwargs: {"dataset_id": kwargs.get("dataset_id")},
    )
    _patch_attr(
        monkeypatch,
        "automl.cli.data",
        "materialize",
        lambda **kwargs: {"materialized": True, "session": kwargs["session"] is active},
    )

    _patch_attr(
        monkeypatch, "automl.cli.eval", "list_eval_datasets", lambda **kwargs: [{"id": "ev_1"}]
    )
    _patch_attr(
        monkeypatch,
        "automl.cli.eval",
        "evaluate",
        lambda **kwargs: {"label": kwargs["label"], "session": kwargs["session"] is active},
    )

    _patch_attr(
        monkeypatch,
        "automl.cli._validate_actions",
        "validate_project",
        lambda **kwargs: ValidationReport(),
    )

    def fake_validate_model(*args, **kwargs):
        validate_model_calls.append({"args": args, "kwargs": kwargs})
        return ValidationReport()

    _patch_attr(monkeypatch, "automl.cli._validate_actions", "validate_model", fake_validate_model)
    _patch_attr(
        monkeypatch,
        "automl.cli._validate_actions",
        "validate_proposal",
        lambda **kwargs: ValidationReport(),
    )
    (tmp_path / "model.py").write_text("class Model:\n    pass\n")

    commands = [
        ["--project", "demo", "--project-root", str(tmp_path), "experiment", "list"],
        ["--project", "demo", "--project-root", str(tmp_path), "experiment", "leaderboard"],
        [
            "--project",
            "demo",
            "--project-root",
            str(tmp_path),
            "experiment",
            "compare",
            "run-a",
            "run-b",
        ],
        ["--project", "demo", "--project-root", str(tmp_path), "experiment", "summary"],
        ["--project", "demo", "--project-root", str(tmp_path), "experiment", "delete", "exp"],
        [
            "--project",
            "demo",
            "--project-root",
            str(tmp_path),
            "--dry-run",
            "experiment",
            "run",
            "--json",
        ],
        ["--project", "demo", "--project-root", str(tmp_path), "experiment", "proposer-context"],
        ["--project", "demo", "--project-root", str(tmp_path), "trial", "list"],
        [
            "--project",
            "demo",
            "--project-root",
            str(tmp_path),
            "trial",
            "create",
            "trial_one",
            "--strategy",
            "baseline",
            "--hypothesis",
            "Try a clean baseline.",
        ],
        [
            "--project",
            "demo",
            "--project-root",
            str(tmp_path),
            "trial",
            "fork",
            "trial_two",
            "--seed",
            "best",
        ],
        [
            "--project",
            "demo",
            "--project-root",
            str(tmp_path),
            "trial",
            "promote",
            "trial_three",
            "--model-path",
            str(tmp_path / "model.py"),
            "--hypothesis",
            "Promote a reviewed file.",
        ],
        ["--project", "demo", "--project-root", str(tmp_path), "trial", "run", "demo"],
        ["--project", "demo", "--project-root", str(tmp_path), "trial", "show", "run-1"],
        ["--project", "demo", "--project-root", str(tmp_path), "trial", "delete", "run-1"],
        ["--project", "demo", "--project-root", str(tmp_path), "trial", "lock", "acquire"],
        [
            "--project",
            "demo",
            "--project-root",
            str(tmp_path),
            "trial",
            "lock",
            "release",
            "--lock-id",
            "lock-1",
        ],
        ["--project", "demo", "--project-root", str(tmp_path), "data", "list"],
        ["--project", "demo", "--project-root", str(tmp_path), "data", "profile"],
        ["--project", "demo", "--project-root", str(tmp_path), "data", "materialize"],
        ["--project", "demo", "--project-root", str(tmp_path), "eval", "list"],
        [
            "--project",
            "demo",
            "--project-root",
            str(tmp_path),
            "eval",
            "compute",
            "--model-run-id",
            "run-1",
            "--eval-dataset",
            "ev_1",
            "--label",
            "test",
        ],
        ["--project", "demo", "--project-root", str(tmp_path), "validate", "project"],
        [
            "--project",
            "demo",
            "--project-root",
            str(tmp_path),
            "validate",
            "model",
            "--module",
            "json",
            "--class-name",
            "JSONDecoder",
        ],
        [
            "--project",
            "demo",
            "--project-root",
            str(tmp_path),
            "validate",
            "proposal",
            "--proposal-json",
            "-",
        ],
    ]

    monkeypatch.setattr("sys.stdin", types.SimpleNamespace(read=lambda: "{}"))
    for command in commands:
        assert main(command) == 0, command
        assert capsys.readouterr().out.strip()

    assert run_calls
    assert validate_model_calls
    assert validate_model_calls[0]["kwargs"]["session"] is active


def test_data_materialize_prints_dataset_manifest_not_loaded_rows(monkeypatch, tmp_path, capsys):
    from automl.cli import main
    from automl.data import ComponentHashes, Dataset

    active = types.SimpleNamespace(
        active_experiment_id="exp",
        config=types.SimpleNamespace(repo_root=tmp_path),
    )
    _patch_use_project(monkeypatch, active)

    def fake_materialize(**kwargs):
        assert kwargs["include_rows"] is False
        assert kwargs["session"] is active
        return Dataset(
            id="v1_manifest",
            identity_hash="sha256:identity",
            component_hashes=ComponentHashes(
                source_identity="sha256:source",
                feature_registry="sha256:registry",
                data_content="sha256:data",
                schema="sha256:schema",
            ),
            gcs_bucket="automl-test-bucket",
            gcs_prefix="",
            project_name="demo",
            created_at="2026-05-27T00:00:00+00:00",
            source_identity={"kind": "local_csv"},
            n_rows=1,
            n_columns=2,
            target_column="target",
            split_id_col="SPLITID",
            hash_key=("row_id",),
        )

    _patch_attr(monkeypatch, "automl.cli.data", "materialize", fake_materialize)

    assert main(["--project", "demo", "--project-root", str(tmp_path), "data", "materialize"]) == 0

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["id"] == "v1_manifest"
    assert payload["n_rows"] == 1
    assert "secret_row_value" not in output


def test_trial_lock_cli_uses_runner_session_lock(monkeypatch, tmp_path, capsys):
    from automl.cli import main

    active = types.SimpleNamespace(
        config=types.SimpleNamespace(repo_root=tmp_path),
        active_experiment_id="cli-catalog",
        project_name="demo",
        dry_run=True,
        namespace="qa",
    )
    _patch_use_project(monkeypatch, active)
    calls = []
    _patch_attr(
        monkeypatch,
        "automl.cli._trial_actions",
        "trial_lock",
        types.SimpleNamespace(
            acquire_for_session=lambda active, session_id: (
                calls.append(("acquire_for_session", {"active": active, "session_id": session_id}))
                or {
                    "status": "acquired",
                    "session_id": session_id,
                    "route": "qa/dry_run/demo/cli-catalog",
                    "lock_id": "lock-1",
                }
            ),
            release_for_session=lambda active, session_id, lock_id: (
                calls.append(
                    (
                        "release_for_session",
                        {"active": active, "session_id": session_id, "lock_id": lock_id},
                    )
                )
                or {"status": "released", "session_id": session_id, "lock_id": lock_id}
            ),
        ),
    )

    assert (
        main(
            [
                "--project",
                "demo",
                "--project-root",
                str(tmp_path),
                "--dry-run",
                "--namespace",
                "qa",
                "trial",
                "lock",
                "acquire",
                "--session-id",
                "session-1",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {
        "status": "acquired",
        "session_id": "session-1",
        "route": "qa/dry_run/demo/cli-catalog",
        "lock_id": "lock-1",
    }

    assert (
        main(
            [
                "--project",
                "demo",
                "--project-root",
                str(tmp_path),
                "trial",
                "lock",
                "release",
                "--session-id",
                "session-1",
                "--lock-id",
                "lock-1",
            ]
        )
        == 0
    )

    assert calls == [
        (
            "acquire_for_session",
            {
                "active": active,
                "session_id": "session-1",
            },
        ),
        (
            "release_for_session",
            {
                "active": active,
                "session_id": "session-1",
                "lock_id": "lock-1",
            },
        ),
    ]


def test_trial_create_can_resolve_slug_and_strategy_from_proposal_json(
    monkeypatch, tmp_path, capsys
):
    from automl.cli import main

    active = types.SimpleNamespace(
        active_experiment_id="cli-catalog",
        config=types.SimpleNamespace(repo_root=tmp_path),
    )
    _patch_use_project(monkeypatch, active)
    calls = []
    proposal_path = tmp_path / "proposal.json"
    proposal_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "slug": "proposal_trial",
                "strategy": "baseline",
                "hypothesis": "Use proposal metadata.",
                "seed_hint": "auto",
                "implementation_plan": ["fit"],
                "constraints": ["safe"],
                "required_dependencies": ["pandas"],
            }
        ),
        encoding="utf-8",
    )

    def fake_create_trial(**kwargs):
        calls.append(kwargs)
        return tmp_path / "proposal_trial"

    _patch_attr(monkeypatch, "automl.cli._trial_actions", "create_trial", fake_create_trial)

    assert (
        main(
            [
                "--project",
                "demo",
                "--project-root",
                str(tmp_path),
                "trial",
                "create",
                "--proposal-json",
                str(proposal_path),
            ]
        )
        == 0
    )

    assert json.loads(capsys.readouterr().out) == {"trial_dir": str(tmp_path / "proposal_trial")}
    assert calls[0]["slug"] is None
    assert calls[0]["strategy"] is None
    assert calls[0]["hypothesis"] == ""
    assert calls[0]["seed"] is None
    assert calls[0]["proposal"]["slug"] == "proposal_trial"
    assert calls[0]["session"] is active


def test_experiment_run_builds_dry_run_launch_from_top_level_flags(monkeypatch, tmp_path, capsys):
    from automl.agent.launch import LaunchSpec
    from automl.cli import main

    active = types.SimpleNamespace(
        active_experiment_id="cli-catalog",
        dry_run=True,
        namespace="qa",
        project_name="demo",
    )
    calls = _patch_use_project(monkeypatch, active)
    launch_calls = []
    run_calls = []

    def fake_build_launch(**kwargs):
        launch_calls.append(kwargs)
        assert kwargs["session"] is active
        return LaunchSpec(
            command=["claude-test"],
            env={"AUTOML_INHERIT_DRY_RUN": "1", "AUTOML_NAMESPACE": "qa"},
            cwd=tmp_path,
        )

    _patch_attr(monkeypatch, "automl.cli._experiment_actions", "build_launch", fake_build_launch)
    monkeypatch.setattr(
        "automl.cli._experiment_actions.subprocess.run",
        lambda command, **kwargs: (
            run_calls.append((command, kwargs)) or subprocess.CompletedProcess(command, 0)
        ),
    )

    assert (
        main(
            [
                "--project",
                "demo",
                "--project-root",
                str(tmp_path),
                "--dry-run",
                "--namespace",
                "qa",
                "--experiment-id",
                "cli-catalog",
                "experiment",
                "run",
                "--json",
            ]
        )
        == 0
    )

    assert calls[0]["dry_run"] is True
    assert calls[0]["namespace"] == "qa"
    assert launch_calls[0]["session"] is active
    assert run_calls[0][1]["env"]["AUTOML_INHERIT_DRY_RUN"] == "1"
    assert json.loads(capsys.readouterr().out)["returncode"] == 0


def test_experiment_run_forwards_loop_options_to_skill_command(monkeypatch, tmp_path):
    from automl.agent.launch import LaunchSpec
    from automl.cli import main

    active = types.SimpleNamespace(
        active_experiment_id="cli-catalog",
        dry_run=False,
        namespace="",
        project_name="demo",
    )
    _patch_use_project(monkeypatch, active)
    launch_calls = []

    def fake_build_launch(**kwargs):
        launch_calls.append(kwargs)
        return LaunchSpec(command=["claude-test"], env={}, cwd=tmp_path)

    _patch_attr(monkeypatch, "automl.cli._experiment_actions", "build_launch", fake_build_launch)
    monkeypatch.setattr(
        "automl.cli._experiment_actions.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0),
    )

    assert (
        main(
            [
                "--project",
                "demo",
                "--project-root",
                str(tmp_path),
                "experiment",
                "run",
                "--max-iter",
                "7",
                "--time-budget",
                "1.5",
                "--instruction",
                "prefer linear models",
            ]
        )
        == 0
    )

    assert launch_calls[0]["automl_args"] == [
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
    ]


def test_experiment_run_forwards_root_route_flags_to_skill_command(monkeypatch, tmp_path):
    from automl.agent.launch import LaunchSpec
    from automl.cli import main

    active = types.SimpleNamespace(
        active_experiment_id="cli-catalog",
        dry_run=True,
        namespace="qa",
        project_name="demo",
    )
    _patch_use_project(monkeypatch, active)
    launch_calls = []

    def fake_build_launch(**kwargs):
        launch_calls.append(kwargs)
        return LaunchSpec(command=["claude-test"], env={}, cwd=tmp_path)

    _patch_attr(monkeypatch, "automl.cli._experiment_actions", "build_launch", fake_build_launch)
    monkeypatch.setattr(
        "automl.cli._experiment_actions.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0),
    )

    assert (
        main(
            [
                "--project",
                "demo",
                "--project-root",
                str(tmp_path),
                "--dry-run",
                "--namespace",
                "qa",
                "experiment",
                "run",
                "--max-iter",
                "7",
            ]
        )
        == 0
    )

    assert launch_calls[0]["automl_args"] == [
        "experiment",
        "run",
        "--project",
        "demo",
        "--dry-run",
        "--namespace",
        "qa",
        "--max-iter",
        "7",
    ]


def test_experiment_run_forwards_refresh_and_confirmation_flags(monkeypatch, tmp_path):
    from automl.agent.launch import LaunchSpec
    from automl.cli import main

    active = types.SimpleNamespace(
        active_experiment_id="cli-catalog",
        dry_run=True,
        namespace="qa",
        project_name="demo",
    )
    _patch_use_project(monkeypatch, active)
    launch_calls = []

    def fake_build_launch(**kwargs):
        launch_calls.append(kwargs)
        return LaunchSpec(command=["claude-test"], env={}, cwd=tmp_path)

    _patch_attr(monkeypatch, "automl.cli._experiment_actions", "build_launch", fake_build_launch)
    monkeypatch.setattr(
        "automl.cli._experiment_actions.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0),
    )

    assert (
        main(
            [
                "--project",
                "demo",
                "--project-root",
                str(tmp_path),
                "--dry-run",
                "--namespace",
                "qa",
                "experiment",
                "run",
                "--max-iter",
                "7",
                "--time-budget",
                "1.5",
                "--refresh-source",
                "--auto-confirm",
                "--instruction",
                "prefer linear models",
            ]
        )
        == 0
    )

    assert launch_calls[0]["automl_args"] == [
        "experiment",
        "run",
        "--project",
        "demo",
        "--dry-run",
        "--namespace",
        "qa",
        "--max-iter",
        "7",
        "--time-budget",
        "1.5",
        "--refresh-source",
        "--auto-confirm",
        "--instruction",
        "prefer linear models",
    ]


def test_delete_wrappers_use_root_session_flags(monkeypatch, tmp_path, capsys):
    from automl.cli import main

    active = types.SimpleNamespace(
        active_experiment_id="cli-catalog",
        dry_run=True,
        namespace="qa",
    )
    _patch_use_project(monkeypatch, active)
    calls = []

    def record_delete(kind):
        def fake_delete(identifier, **kwargs):
            calls.append(
                {
                    "kind": kind,
                    "identifier": identifier,
                    "dry_run": kwargs["session"].dry_run,
                    "namespace": kwargs["session"].namespace,
                    "apply": kwargs["apply"],
                }
            )
            return calls[-1]

        return fake_delete

    _patch_attr(monkeypatch, "automl.cli.project", "delete_project", record_delete("project"))
    _patch_attr(
        monkeypatch,
        "automl.cli._experiment_actions",
        "delete_experiment",
        record_delete("experiment"),
    )
    _patch_attr(monkeypatch, "automl.cli._trial_actions", "delete_trial", record_delete("trial"))

    base = ["--project", "demo", "--project-root", str(tmp_path), "--dry-run", "--namespace", "qa"]
    assert main([*base, "project", "delete", "demo", "--apply"]) == 0
    assert main([*base, "experiment", "delete", "cli-catalog", "--apply"]) == 0
    assert main([*base, "trial", "delete", "run-1", "--apply"]) == 0
    capsys.readouterr()

    assert calls == [
        {
            "kind": "project",
            "identifier": "demo",
            "dry_run": True,
            "namespace": "qa",
            "apply": True,
        },
        {
            "kind": "experiment",
            "identifier": "cli-catalog",
            "dry_run": True,
            "namespace": "qa",
            "apply": True,
        },
        {
            "kind": "trial",
            "identifier": "run-1",
            "dry_run": True,
            "namespace": "qa",
            "apply": True,
        },
    ]


@pytest.mark.parametrize(
    "command",
    [
        ["--project", "demo", "experiment", "delete", "cli-catalog", "--dry-run"],
        ["--project", "demo", "experiment", "delete", "cli-catalog", "--route", "dry_run"],
        ["--project", "demo", "experiment", "delete", "cli-catalog", "--route-namespace", "qa"],
        ["--project", "demo", "trial", "delete", "run-1", "--dry-run"],
        ["--project", "demo", "project", "delete", "demo", "--route-namespace", "qa"],
    ],
)
def test_delete_verbs_reject_per_verb_routing_flags(command):
    from automl.cli import main

    with pytest.raises(SystemExit):
        main(command)


@pytest.mark.parametrize("verb", ["run", "inspect", "loop-context", "profile", "propose"])
def test_retired_top_level_verbs_are_not_registered(verb):
    from automl.cli import main

    with pytest.raises(SystemExit):
        main([verb])
