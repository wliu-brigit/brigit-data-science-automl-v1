from __future__ import annotations

import copy
from pathlib import Path

import pytest

from tests.e2e._gates import requires_live_e2e

from automl.data import (
    DataSpec,
    GCSParquetSource,
    SnowflakeSource,
    list_datasets,
    load_dataset_by_id,
    load_dataset_by_trial,
    materialize,
    profile,
)
from automl.mlflow import client as mlflow_client
from automl.mlflow import tags
from automl.project import clear_session, use_project
from automl.runner import run_trial
from automl.trial import TrialStatus

pytestmark = [pytest.mark.e2e, pytest.mark.qa]


@requires_live_e2e("Home Credit data model")
def test_homecredit_data_model_breadth_external_gate():
    repo_root = Path(__file__).resolve().parents[2]
    active = use_project("example_homecredit", repo_root=repo_root)
    try:
        loaded = materialize(session=active)
        index = list_datasets(session=active)

        assert loaded.dataset in index.datasets
        assert index.active.id == loaded.id
        assert active.config.data_spec.source.kind == "local_csv"
        assert (
            GCSParquetSource("gs://bucket/path/train.parquet", hash_key="row_id").identity()["kind"]
            == "gcs_parquet"
        )
        assert (
            DataSpec(
                source=SnowflakeSource(
                    base_table="APP",
                    base_data_sql="sql/base.sql",
                    training_data_sql="sql/train.sql",
                )
            ).source.kind
            == "snowflake"
        )

        registry_frame = loaded.registry.to_dataframe()
        assert "golden" not in registry_frame.columns
        assert "weak" not in registry_frame.columns
        trial_registry = copy.deepcopy(loaded.registry)
        trial_registry.add_derived(
            "homecredit_credit_log",
            "num",
            ("amt_credit",),
        )
        assert trial_registry.get("homecredit_credit_log").derived is True
        assert trial_registry.get("homecredit_credit_log").source_columns == ("amt_credit",)

        multi = load_dataset_by_id(
            loaded.id,
            split_range=((80, 90), (95, 100)),
            session=active,
        )
        assert multi.split_name is None
        assert multi.split_ranges == ((80, 90), (95, 100))
        assert set(multi.df["SPLIT_PCT"]).issubset(set(range(80, 90)) | set(range(95, 100)))

        result = run_trial("example_homecredit", session=active)

        assert result.status == TrialStatus.FINISHED.value
        assert result.run_id
        assert result.metrics["auc"] >= 0.0
        run = mlflow_client.raw().get_run(result.run_id)
        _assert_trial_artifact_exists(result.run_id, run.data.tags[tags.DATA_CONTRACT_URI])
        _assert_runs_uri_exists(run.data.tags[tags.MODEL_URI])
        _assert_trial_artifact_exists(result.run_id, run.data.tags[tags.eval_uri("test")])
        _assert_trial_artifact_exists(result.run_id, run.data.tags[tags.eval_uri("train")])
        assert run.data.tags[tags.TRIAL_ID] == result.trial_id

        profiled = profile(session=active)
        assert profiled.dataset_id == loaded.id
        assert profiled.chart_uris
        _assert_runs_uri_exists(profiled.data_card_uri)
        _assert_runs_uri_exists(profiled.data_observations_uri)
        _assert_runs_uri_exists(profiled.profile_manifest_uri)

        replayed = load_dataset_by_trial(
            result.trial_id,
            split_name=active.config.run_config.train_split,
            session=active,
        )
        assert replayed.id == loaded.id
        assert replayed.split_name == active.config.run_config.train_split
    finally:
        clear_session()


def _assert_runs_uri_exists(uri: str) -> None:
    assert uri.startswith("runs:/")
    _, remainder = uri.split("runs:/", 1)
    run_id, artifact_path = remainder.strip("/").split("/", 1)
    _assert_trial_artifact_exists(run_id, artifact_path)


def _assert_trial_artifact_exists(run_id: str, artifact_path: str) -> None:
    local_path = mlflow_client.raw().download_artifacts(run_id, artifact_path)
    assert Path(local_path).exists()
