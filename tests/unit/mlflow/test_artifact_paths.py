"""Tests for MLflow artifact path helpers."""

from __future__ import annotations

import pytest

from automl.mlflow.artifact_paths import json_artifact_path

pytestmark = pytest.mark.unit


def test_json_artifact_path_preserves_nested_path_and_appends_suffix() -> None:
    assert json_artifact_path("debug/sample") == "debug/sample.json"


def test_json_artifact_path_strips_outer_slashes_without_duplicate_suffix() -> None:
    assert json_artifact_path("/debug/sample.json/") == "debug/sample.json"


@pytest.mark.parametrize("name", ["", "/", "///"])
def test_json_artifact_path_requires_non_empty_name(name: str) -> None:
    with pytest.raises(ValueError, match="^JSON artifact name required$"):
        json_artifact_path(name)
