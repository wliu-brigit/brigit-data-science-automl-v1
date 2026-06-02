import importlib
import json

import pandas as pd
import pytest

from automl.data import ComponentHashes, Dataset, FeatureRegistry, LoadedDataset, Profile
from automl.data.profile import _write_profile_artifacts
from automl.utils.hashing import dataframe_content_hash, schema_hash

pytestmark = pytest.mark.unit


def _loaded() -> LoadedDataset:
    df = pd.DataFrame(
        {
            "row_id": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
            "amount": [10.0, None, 30.0, 40.0],
            "segment": ["a", "b", "a", "c"],
            "SPLITID": [10, 20, 90, 95],
        }
    )
    registry = FeatureRegistry().build_from_df(
        df,
        target_column="target",
        metadata_cols=("row_id",),
        split_id_col="SPLITID",
    )
    dataset = Dataset(
        id="v1_profile",
        identity_hash="sha256:identity",
        component_hashes=ComponentHashes(
            source_identity="sha256:source",
            feature_registry=registry.content_hash(),
            data_content=dataframe_content_hash(df),
            schema=schema_hash(df),
        ),
        gcs_bucket="automl-test-bucket",
        gcs_prefix="",
        project_name="demo",
        created_at="2026-05-27T00:00:00+00:00",
        source_identity={"kind": "local_csv"},
        n_rows=len(df),
        n_columns=len(df.columns),
        target_column="target",
        split_id_col="SPLITID",
        hash_key=("row_id",),
    )
    return LoadedDataset(dataset=dataset, df=df, registry=registry)


def test_profile_round_trips_and_strips_unknown_keys():
    profile = Profile(
        dataset_id="v1_profile",
        target_column="target",
        data_card_uri="runs:/run/v1_profile/profile/data_card.json",
        data_observations_uri="runs:/run/v1_profile/profile/data_observations.json",
        profile_manifest_uri="runs:/run/v1_profile/profile/profile_manifest.json",
        chart_uris={"label_distribution": "runs:/run/v1_profile/profile/charts/label.png"},
        created_at="2026-05-27T00:00:00+00:00",
    )

    restored = Profile.from_dict({**profile.to_dict(), "unknown": "ignored"})

    assert restored == profile
    assert restored.schema_version == 1


def test_write_profile_artifacts_produces_card_observations_manifest_and_charts(tmp_path):
    written = _write_profile_artifacts(_loaded(), tmp_path)

    assert (tmp_path / "data_card.json").exists()
    assert (tmp_path / "data_observations.json").exists()
    assert (tmp_path / "profile_manifest.json").exists()
    assert any((tmp_path / "charts").glob("*.png"))
    assert written["dataset_id"] == "v1_profile"
    assert written["target_column"] == "target"
    card = json.loads((tmp_path / "data_card.json").read_text())
    assert card["n_rows"] == 4
    assert card["target"] == "target"


def test_profile_artifact_writer_wraps_crashing_chart_as_observation(monkeypatch, tmp_path):
    profile_module = importlib.import_module("automl.data.profile")

    def crashing_chart(df, target, out_path):
        raise RuntimeError("chart exploded")

    monkeypatch.setattr(
        profile_module,
        "_CHARTS",
        [("broken_chart", crashing_chart), *profile_module._CHARTS],
    )

    _write_profile_artifacts(_loaded(), tmp_path)

    observations = json.loads((tmp_path / "data_observations.json").read_text())
    assert any("broken_chart" in row["text"] for row in observations["observations"])
