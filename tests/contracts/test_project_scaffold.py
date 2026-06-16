"""Ratchet: the project scaffold's shape is a published convention.

Projects mirror the library's domains (eval/, model/, data/, tests/) with a
README front door — see automl/project/templates/README.md. Changing the
scaffold shape is fine; update this contract in the same change.
"""

import pytest

from automl.project.scaffold import create_project

pytestmark = pytest.mark.contract

EXPECTED_SCAFFOLD = [
    "projects/shape_check/PROJECT_INSTRUCTIONS.md",
    "projects/shape_check/README.md",
    "projects/shape_check/__init__.py",
    "projects/shape_check/config.py",
    "projects/shape_check/data/__init__.py",
    "projects/shape_check/data/queries/base_table.sql",
    "projects/shape_check/data/queries/training_data.sql",
    "projects/shape_check/docs/experiment-learnings.md",
    "projects/shape_check/eval/__init__.py",
    "projects/shape_check/model/__init__.py",
    "projects/shape_check/tests/__init__.py",
]


def test_project_init_creates_the_published_shape(tmp_path):
    out = create_project("shape_check", project_root=tmp_path)
    assert sorted(out["created"]) == EXPECTED_SCAFFOLD


def test_scaffolded_readme_carries_the_conventions(tmp_path):
    create_project("shape_check", project_root=tmp_path)
    readme = (tmp_path / "projects" / "shape_check" / "README.md").read_text(encoding="utf-8")
    # The sections every project relies on; renames here are doc-breaking.
    assert "## Layout — mirror the library's domains" in readme
    assert "## Project-specific dependencies" in readme
    assert "uv add --group shape_check" in readme  # group named after the project
    assert "## Writing PROJECT_INSTRUCTIONS.md" in readme
    assert "## Snowflake: when your base table already exists" in readme
    assert "{project_name}" not in readme  # substitution applied
    # Progressive-disclosure entry point into docs/ (experiment-learnings guide)
    assert "## Project docs" in readme
    assert "docs/experiment-learnings.md" in readme


def test_scaffolded_docs_carry_the_experiment_learnings_guide(tmp_path):
    create_project("shape_check", project_root=tmp_path)
    guide = (
        tmp_path / "projects" / "shape_check" / "docs" / "experiment-learnings.md"
    ).read_text(encoding="utf-8")
    assert "# Publishing an experiment-level learning" in guide
    assert "schema_version: 1" in guide  # versioned study format
    assert "MLflow is the durable home" in guide
