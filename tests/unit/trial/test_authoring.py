from __future__ import annotations

import importlib
import json
import linecache
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


class NotebookModel:
    def fit(self, df_train, registry, seed=0):
        return self


def _write_minimal_project(root: Path, name: str = "demo") -> None:
    project_dir = root / "projects" / name
    project_dir.mkdir(parents=True)
    (root / "projects" / "__init__.py").write_text("")
    (project_dir / "__init__.py").write_text("")
    (project_dir / "config.py").write_text(
        "\n".join(
            [
                "from automl.project import ModelRoute, ModelsConfig, ProjectConfig, RunConfig",
                "PROJECT_CONFIG = ProjectConfig.partial(",
                "    run_config=RunConfig(",
                "        experiment_id='exp',",
                "        models=ModelsConfig(",
                "            manager=ModelRoute(model='manager', effort='low'),",
                "            proposer=ModelRoute(model='proposer', effort='low'),",
                "            coder=ModelRoute(model='coder', effort='low'),",
                "        ),",
                "        per_trial_seconds=60,",
                "    ),",
                ")",
                "",
            ]
        )
    )


def test_create_writes_routed_trial_folder_metadata_source_and_proposal(tmp_path):
    from automl.project import clear_session, use_project
    from automl.trial.create import create

    _write_minimal_project(tmp_path)
    active = use_project(
        "demo",
        repo_root=tmp_path,
        dry_run=True,
        namespace="qa",
        experiment_id="exp",
    )
    model_source = tmp_path / "candidate_model.py"
    model_source.write_text("class Model:\n    pass\n\nMODEL_CLASS = Model\n")

    try:
        trial_path = create(
            slug="trial_one",
            strategy="logistic_baseline",
            hypothesis="Regularized logistic regression is a clean baseline.",
            model_source=model_source,
            proposal={"schema_version": 2, "slug": "trial_one"},
            session=active,
        )
    finally:
        clear_session()

    assert trial_path == (
        tmp_path
        / "projects"
        / "demo"
        / "experiments"
        / "qa"
        / "dry_run"
        / "demo"
        / "exp"
        / "trial_one"
    )
    assert (trial_path / "run.py").is_file()
    run_text = (trial_path / "run.py").read_text(encoding="utf-8")
    assert "AUTOML_STATUS=" in run_text
    assert "AUTOML_TRIAL_ID=" in run_text
    assert "AUTOML_RUN_ID=" in run_text
    assert "AUTOML_ERROR=" in run_text
    assert (trial_path / "model.py").read_text() == model_source.read_text()
    assert json.loads((trial_path / "proposal" / "proposal.json").read_text()) == {
        "schema_version": 2,
        "slug": "trial_one",
    }

    metadata = json.loads((trial_path / "metadata.json").read_text())
    assert metadata["schema_version"] == 1
    assert metadata["slug"] == "trial_one"
    assert metadata["strategy"] == "logistic_baseline"
    assert metadata["hypothesis"] == "Regularized logistic regression is a clean baseline."
    assert metadata["training_origin"] == "automl"
    assert metadata["project_name"] == "demo"
    assert metadata["project_package"] == "projects.demo"
    assert metadata["experiment_id"] == "exp"
    assert "trial_id" not in metadata
    assert "dry_run" not in metadata
    assert "run_mode" not in metadata


def test_create_resolves_proposal_defaults(monkeypatch):
    create_module = importlib.import_module("automl.trial.create")
    proposal = {
        "slug": "proposal_trial",
        "strategy": "baseline",
        "hypothesis": "Use proposal metadata.",
        "seed_hint": "auto",
    }
    calls = []

    def fake_create_resolved(**kwargs):
        calls.append(kwargs)
        return Path("proposal_trial")

    monkeypatch.setattr(create_module, "_create_resolved", fake_create_resolved)

    trial_path = create_module.create(proposal=proposal, session=object())

    assert trial_path == Path("proposal_trial")
    assert calls[0]["slug"] == "proposal_trial"
    assert calls[0]["strategy"] == "baseline"
    assert calls[0]["hypothesis"] == "Use proposal metadata."
    assert calls[0]["seed"] == "auto"


def test_package_model_writes_class_source_and_model_alias(tmp_path):
    from automl.trial.packaging import package_model

    target = package_model(
        NotebookModel,
        imports=["from __future__ import annotations"],
        output_path=tmp_path / "model.py",
    )

    text = target.read_text()
    assert text.startswith("from __future__ import annotations\n\n")
    assert "class NotebookModel:" in text
    assert "def fit(self, df_train, registry, seed=0):" in text
    assert text.rstrip().endswith("Model = NotebookModel")


def test_package_model_falls_back_to_notebook_cell_source(tmp_path):
    from automl.trial.packaging import package_model

    filename = "<notebook-cell-package-model>"
    source = "\n".join(
        [
            "class NotebookCellModel:",
            "    name = 'notebook_cell_model'",
            "",
            "    def fit(self, df_train, registry, seed=0):",
            "        return self",
            "",
        ]
    )
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace = {"__name__": "__notebook_cell__"}

    try:
        exec(compile(source, filename, "exec"), namespace)
        target = package_model(
            namespace["NotebookCellModel"],
            imports=[],
            output_path=tmp_path / "model.py",
        )
    finally:
        linecache.cache.pop(filename, None)

    text = target.read_text()
    assert "class NotebookCellModel:" in text
    assert "def fit(self, df_train, registry, seed=0):" in text
    assert text.rstrip().endswith("Model = NotebookCellModel")


def test_fork_creates_human_trial_from_seed(monkeypatch, tmp_path):
    fork_module = importlib.import_module("automl.trial.fork")

    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        return tmp_path / "forked_trial"

    monkeypatch.setattr(fork_module, "create", fake_create)

    trial_path = fork_module.fork(
        slug="human_fork",
        seed="best",
        strategy="manual_fork",
        hypothesis="Try a hand-tuned variant.",
        session="active-session",
    )

    assert trial_path == tmp_path / "forked_trial"
    assert calls == [
        {
            "slug": "human_fork",
            "strategy": "manual_fork",
            "hypothesis": "Try a hand-tuned variant.",
            "seed": "best",
            "training_origin": "human",
            "session": "active-session",
        }
    ]


def test_trial_domain_does_not_export_promote_or_import_runner():
    trial = importlib.import_module("automl.trial")

    assert "promote" not in trial.__all__
    assert not hasattr(trial, "promote")


def test_trial_domain_facade_does_not_export_create_request_helper():
    trial = importlib.import_module("automl.trial")

    assert "create_from_request" not in trial.__all__
    assert not hasattr(trial, "create_from_request")
