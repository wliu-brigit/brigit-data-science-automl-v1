"""Recipe: the config-derived identity of a materialization."""

from pathlib import Path

import pytest

from automl.data import DataSpec, LocalCSVSource
from automl.data.recipe import compute_recipe, recipe_diff
from automl.project import BinaryClassification, ProjectConfig, Session

pytestmark = pytest.mark.unit


def _spec_and_session(tmp_path, csv_name="a.csv", exclude=(), dry_run=False):
    spec = DataSpec(
        source=LocalCSVSource(csv_path=tmp_path / csv_name, unique_key="row_id"),
        exclude_cols=tuple(exclude),
    )
    config = ProjectConfig(
        project_name="demo",
        project_dir=Path(tmp_path),
        task=BinaryClassification(target="target"),
        data_spec=spec,
    )
    return spec, Session(config=config, dry_run=dry_run)


def test_recipe_is_computed_without_touching_the_source(tmp_path):
    spec, session = _spec_and_session(tmp_path, csv_name="missing.csv")
    recipe = compute_recipe(spec, session)  # file does not exist; must not be read
    assert recipe["source"]["kind"] == "local_csv"
    assert recipe["target"]
    assert "dry_run_rows" not in recipe


def test_recipe_canonicalizes_cosmetic_ordering(tmp_path):
    spec_a, session = _spec_and_session(tmp_path, exclude=("b", "a"))
    spec_b, _ = _spec_and_session(tmp_path, exclude=("a", "b"))
    assert compute_recipe(spec_a, session) == compute_recipe(spec_b, session)


def test_recipe_includes_dry_run_rows_only_in_dry_run_sessions(tmp_path):
    spec, session = _spec_and_session(tmp_path, dry_run=True)
    assert compute_recipe(spec, session)["dry_run_rows"] == spec.dry_run_rows


def test_recipe_diff_names_changed_fields_with_dotted_paths():
    recorded = {"target": "y", "source": {"kind": "local_csv", "csv_path": "a.csv"}}
    current = {"target": "y", "source": {"kind": "local_csv", "csv_path": "b.csv"}}
    assert recipe_diff(recorded, current) == ["source.csv_path"]
    assert recipe_diff(recorded, recorded) == []
