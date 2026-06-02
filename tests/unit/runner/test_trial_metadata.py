from __future__ import annotations

import json
from pathlib import Path

import pytest

from automl.errors import ProjectError
from automl.trial.metadata import TrialMetadata

pytestmark = pytest.mark.unit


def _write_metadata(trial_dir: Path, payload: object) -> Path:
    trial_dir.mkdir()
    metadata_path = trial_dir / "metadata.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    return metadata_path


def test_read_trial_metadata_returns_domain_schema(tmp_path):
    from automl.runner.trial import _read_trial_metadata

    trial_dir = tmp_path / "trial_one"
    _write_metadata(
        trial_dir,
        {
            "schema_version": 1,
            "slug": "trial_one",
            "strategy": "wide_linear",
            "hypothesis": "A wider linear model may improve calibration.",
            "training_origin": "human",
            "project_name": "demo",
            "project_package": "projects.demo",
            "experiment_id": "exp",
        },
    )

    metadata = _read_trial_metadata(trial_dir)

    assert isinstance(metadata, TrialMetadata)
    assert metadata.slug == "trial_one"
    assert metadata.strategy == "wide_linear"
    assert metadata.hypothesis == "A wider linear model may improve calibration."
    assert metadata.training_origin == "human"
    assert metadata.project_name == "demo"
    assert metadata.project_package == "projects.demo"
    assert metadata.experiment_id == "exp"


def test_read_trial_metadata_missing_file_raises_project_error(tmp_path):
    from automl.runner.trial import _read_trial_metadata

    trial_dir = tmp_path / "missing"

    with pytest.raises(
        ProjectError,
        match=f"trial metadata not found at {trial_dir / 'metadata.json'}",
    ):
        _read_trial_metadata(trial_dir)


def test_read_trial_metadata_non_object_json_raises_project_error(tmp_path):
    from automl.runner.trial import _read_trial_metadata

    trial_dir = tmp_path / "trial_one"
    _write_metadata(trial_dir, ["not", "an", "object"])

    with pytest.raises(
        ProjectError,
        match=f"trial metadata must be a JSON object: {trial_dir / 'metadata.json'}",
    ):
        _read_trial_metadata(trial_dir)


def test_read_trial_metadata_invalid_json_raises_project_error(tmp_path):
    from automl.runner.trial import _read_trial_metadata

    trial_dir = tmp_path / "trial_one"
    trial_dir.mkdir()
    metadata_path = trial_dir / "metadata.json"
    metadata_path.write_text("{", encoding="utf-8")

    with pytest.raises(
        ProjectError,
        match=(
            f"invalid trial metadata JSON at {metadata_path}: "
            r"Expecting property name enclosed in double quotes: line 1 column 2 \(char 1\)"
        ),
    ):
        _read_trial_metadata(trial_dir)


def test_read_trial_metadata_missing_optional_fields_default_to_empty_values(tmp_path):
    from automl.runner.trial import _read_trial_metadata

    trial_dir = tmp_path / "trial_one"
    _write_metadata(
        trial_dir,
        {
            "slug": "trial_one",
            "project_name": "demo",
            "project_package": "projects.demo",
            "experiment_id": "exp",
        },
    )

    metadata = _read_trial_metadata(trial_dir)

    assert metadata.strategy == ""
    assert metadata.hypothesis == ""
    assert metadata.training_origin == ""
    assert metadata.created_at == ""
    assert metadata.seed is None
