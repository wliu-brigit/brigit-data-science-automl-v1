from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _session(tmp_path: Path):
    from automl.project import ProjectConfig, Session

    project_dir = tmp_path / "projects" / "demo"
    project_dir.mkdir(parents=True)
    config = ProjectConfig(
        project_name="demo",
        repo_root=tmp_path,
        project_dir=project_dir,
        project_package="projects.demo",
        config_path=project_dir / "config.py",
        instructions_path=project_dir / "AUTOML.md",
    )
    return Session(config=config, dry_run=True, namespace="qa", experiment_id="exp")


def test_trial_folder_execution_context_loads_model_from_verified_route(tmp_path):
    from automl.runner.trial import _execution_context, _load_model_class
    from automl.trial import paths

    active = _session(tmp_path)
    trial_path = paths.trial_dir(active, "trial_one")
    trial_path.mkdir(parents=True)
    (trial_path / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "slug": "trial_one",
                "strategy": "wide_linear",
                "hypothesis": "A wider linear model may improve calibration.",
                "training_origin": "human",
                "project_name": "demo",
                "project_package": "projects.demo",
                "experiment_id": "exp",
            }
        )
    )
    (trial_path / "model.py").write_text(
        "\n".join(
            [
                "class FolderModel:",
                "    name = 'folder_model'",
                "",
                "MODEL_CLASS = FolderModel",
                "",
            ]
        )
    )

    context = _execution_context(trial_path, session=active)

    assert context.session is active
    assert context.trial_dir == trial_path.resolve()
    assert context.metadata.slug == "trial_one"
    assert context.metadata.strategy == "wide_linear"
    assert context.metadata.training_origin == "human"
    assert _load_model_class(context).__name__ == "FolderModel"


def test_trial_folder_loader_supports_standard_module_decorators(tmp_path):
    from automl.runner.trial import _execution_context, _load_model_class
    from automl.trial import paths

    active = _session(tmp_path)
    trial_path = paths.trial_dir(active, "decorated_trial")
    trial_path.mkdir(parents=True)
    (trial_path / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "slug": "decorated_trial",
                "strategy": "wide_linear",
                "training_origin": "human",
                "project_name": "demo",
                "project_package": "projects.demo",
                "experiment_id": "exp",
            }
        )
    )
    (trial_path / "model.py").write_text(
        "\n".join(
            [
                "from dataclasses import dataclass",
                "",
                "@dataclass",
                "class FolderModel:",
                "    name: str = 'folder_model'",
                "",
                "MODEL_CLASS = FolderModel",
                "",
            ]
        )
    )

    context = _execution_context(trial_path, session=active)

    assert _load_model_class(context)().__dict__ == {"name": "folder_model"}


def test_trial_folder_execution_context_rejects_paths_outside_route_root(tmp_path):
    from automl.runner.trial import _execution_context

    active = _session(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(ValueError, match="outside trial route root"):
        _execution_context(outside, session=active)
