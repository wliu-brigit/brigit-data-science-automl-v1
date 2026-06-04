from pathlib import Path

import pandas as pd
import pytest

from automl.data import (
    DataPipeline,
    DatasetRef,
    DataSpec,
    LocalCSVSource,
    SliceContract,
    TrialDataContract,
    TrialRef,
    build_dataset,
)
from automl.data.pipeline import _validate_existing_dataset_matches_candidate, materialize
from automl.errors import DataError
from automl.project import (
    BinaryClassification,
    ModelRoute,
    ModelsConfig,
    ProjectConfig,
    RunConfig,
    Session,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
HOMECREDIT_SAMPLE = (
    REPO_ROOT / "projects" / "example_homecredit" / "data" / "application_train_sample.csv"
)


def _models() -> ModelsConfig:
    route = ModelRoute("sonnet", "medium")
    return ModelsConfig(manager=route, proposer=route, coder=route)


def _session_for(spec: DataSpec) -> Session:
    return Session(
        config=ProjectConfig(
            project_name="example_homecredit",
            repo_root=REPO_ROOT,
            project_dir=REPO_ROOT / "projects" / "example_homecredit",
            config_path=REPO_ROOT / "projects" / "example_homecredit" / "config.py",
            task=BinaryClassification(target="TARGET"),
            data_spec=spec,
            run_config=RunConfig(
                experiment_id="example-homecredit",
                models=_models(),
                per_trial_seconds=120,
            ),
            gcs_bucket="automl-test-bucket",
            gcs_prefix="",
            mlflow_tracking_uri="file:///tmp/mlruns",
        ),
        dry_run=True,
    )


def _route_session(
    tmp_path: Path,
    *,
    gcs_prefix: str,
    namespace: str = "",
    dry_run: bool = False,
    data_spec: DataSpec | None = None,
) -> Session:
    return Session(
        config=ProjectConfig(
            project_name="demo",
            repo_root=tmp_path,
            project_dir=tmp_path / "projects" / "demo",
            config_path=tmp_path / "projects" / "demo" / "config.py",
            task=BinaryClassification(target="target"),
            data_spec=data_spec,
            run_config=RunConfig(
                experiment_id="source-exp",
                models=_models(),
                per_trial_seconds=120,
            ),
            gcs_bucket="automl-test-bucket",
            gcs_prefix=gcs_prefix,
            mlflow_tracking_uri="file:///tmp/mlruns",
        ),
        namespace=namespace,
        dry_run=dry_run,
    )


def _route_dataset(
    tmp_path: Path,
    *,
    gcs_prefix: str,
    namespace: str = "",
    dry_run: bool = False,
):
    csv_path = tmp_path / "route.csv"
    csv_path.write_text("row_id,target,value\n1,0,10\n2,1,20\n", encoding="utf-8")
    spec = DataSpec(source=LocalCSVSource(csv_path=csv_path, unique_key="row_id"))
    active = _route_session(
        tmp_path,
        gcs_prefix=gcs_prefix,
        namespace=namespace,
        dry_run=dry_run,
        data_spec=spec,
    )
    return build_dataset(session=active).dataset


def test_local_csv_source_reads_csv_from_path(tmp_path):
    csv_path = tmp_path / "tiny.csv"
    csv_path.write_text("row_id,target,value\n1,0,10\n2,1,20\n", encoding="utf-8")

    source = LocalCSVSource(csv_path=csv_path, unique_key="row_id")

    df = source.load()

    assert list(df.columns) == ["row_id", "target", "value"]
    assert df.to_dict(orient="records") == [
        {"row_id": 1, "target": 0, "value": 10},
        {"row_id": 2, "target": 1, "value": 20},
    ]
    assert source.identity()["kind"] == "local_csv"


def test_build_dataset_adds_split_pct_and_feature_registry_for_homecredit_sample():
    spec = DataSpec(
        source=LocalCSVSource(csv_path=HOMECREDIT_SAMPLE, unique_key="SK_ID_CURR"),
        metadata_cols=("SK_ID_CURR",),
        dry_run_rows=25,
    )

    loaded = build_dataset(session=_session_for(spec))

    assert loaded.n_rows == 25
    assert loaded.dataset.target_column == "target"
    assert loaded.dataset.unique_key == ("sk_id_curr",)
    assert "SPLIT_PCT" in loaded.df.columns
    assert set(loaded.df["SPLIT_PCT"]).issubset(set(range(100)))
    assert loaded.registry.get("target").target is True
    assert loaded.registry.get("sk_id_curr").model is False
    assert loaded.registry.get("ext_source_1").model is True
    assert loaded.registry.to_dataframe()["name"].is_unique


@pytest.mark.parametrize(
    ("gcs_prefix", "namespace", "dry_run", "expected_base_path"),
    [
        ("root", "", False, "root/demo/source-exp/data/datasets/unmaterialized"),
        ("root", "", True, "root/dry_run/demo/source-exp/data/datasets/unmaterialized"),
        ("root", "qa", True, "root/qa/dry_run/demo/source-exp/data/datasets/unmaterialized"),
        ("", "", False, "demo/source-exp/data/datasets/unmaterialized"),
        ("", "", True, "dry_run/demo/source-exp/data/datasets/unmaterialized"),
        ("", "qa", True, "qa/dry_run/demo/source-exp/data/datasets/unmaterialized"),
    ],
)
def test_dataset_route_uris_pin_current_project_suffix_strip_behavior(
    tmp_path,
    gcs_prefix,
    namespace,
    dry_run,
    expected_base_path,
):
    dataset = _route_dataset(
        tmp_path,
        gcs_prefix=gcs_prefix,
        namespace=namespace,
        dry_run=dry_run,
    )

    assert dataset.gcs_base_path == expected_base_path
    assert dataset.data_gcs_uri == f"gs://automl-test-bucket/{expected_base_path}/data.parquet"
    assert (
        dataset.registry_gcs_uri
        == f"gs://automl-test-bucket/{expected_base_path}/feature_registry.csv"
    )
    assert dataset.manifest_gcs_uri == f"gs://automl-test-bucket/{expected_base_path}/manifest.json"


def test_standardize_columns_keeps_suffix_collisions_unique(tmp_path):
    csv_path = tmp_path / "collisions.csv"
    spec = DataSpec(source=LocalCSVSource(csv_path=csv_path, unique_key="A"))
    pipeline = DataPipeline(spec, _session_for(spec))
    raw = pd.DataFrame(
        {
            "A": [1],
            "A!": [2],
            "A_2": [3],
            "TARGET": [0],
        }
    )

    df, original_names = pipeline.standardize_columns(raw)

    assert list(df.columns) == ["a", "a_2", "a_2_2", "target"]
    assert df.columns.is_unique
    assert original_names["a_2_2"] == "A_2"


def test_build_dataset_resolves_raw_target_column_to_normalized_name(tmp_path):
    csv_path = tmp_path / "raw_target.csv"
    csv_path.write_text("row_id,TARGET,value\n1,0,10\n2,1,20\n", encoding="utf-8")
    spec = DataSpec(source=LocalCSVSource(csv_path=csv_path, unique_key="row_id"), dry_run_rows=10)

    loaded = build_dataset(session=_session_for(spec))

    assert loaded.dataset.target_column == "target"
    assert loaded.registry.get("target").target is True


def test_build_dataset_raises_data_error_when_target_is_missing(tmp_path):
    csv_path = tmp_path / "missing_target.csv"
    csv_path.write_text("row_id,value\n1,10\n2,20\n", encoding="utf-8")
    spec = DataSpec(source=LocalCSVSource(csv_path=csv_path, unique_key="row_id"), dry_run_rows=10)

    with pytest.raises(DataError, match="target column"):
        build_dataset(session=_session_for(spec))


def test_materialize_partial_dataset_objects_raise_data_error(tmp_path, monkeypatch):
    csv_path = tmp_path / "partial.csv"
    csv_path.write_text("row_id,target,value\n1,0,10\n2,1,20\n", encoding="utf-8")
    spec = DataSpec(source=LocalCSVSource(csv_path=csv_path, unique_key="row_id"), dry_run_rows=10)
    active = _route_session(tmp_path, gcs_prefix="", data_spec=spec)

    monkeypatch.setattr(
        "automl.data.pipeline.experiment_artifacts.read_dataset_index",
        lambda: {"datasets": []},
    )
    monkeypatch.setattr(
        "automl.data.pipeline._dataset_object_state",
        lambda dataset: {"data": True, "registry": False, "manifest": False},
    )

    with pytest.raises(DataError, match="partial dataset objects"):
        materialize(session=active)


def test_existing_dataset_identity_mismatch_raises_data_error(tmp_path):
    dataset = _route_dataset(tmp_path, gcs_prefix="")
    candidate = dataset.__class__.from_dict({**dataset.to_dict(), "n_rows": dataset.n_rows + 1})

    with pytest.raises(DataError, match="existing dataset .* does not match candidate fields"):
        _validate_existing_dataset_matches_candidate(dataset, candidate)


def test_quality_filtering_drops_constant_and_high_null_columns_but_preserves_protected(
    tmp_path,
):
    csv_path = tmp_path / "quality.csv"
    pd.DataFrame(
        {
            "row_id": [1, 2, 3],
            "TARGET": [0, 1, 0],
            "metadata_all_null": [None, None, None],
            "strict_constant": [7, 7, 7],
            "mostly_null": [None, None, 3],
            "usable": [1, None, 2],
        }
    ).to_csv(csv_path, index=False)
    spec = DataSpec(
        source=LocalCSVSource(csv_path=csv_path, unique_key="row_id"),
        metadata_cols=("metadata_all_null",),
        null_drop_threshold=0.5,
        constant_drop_threshold=1.0,
        dry_run_rows=10,
    )

    loaded = build_dataset(session=_session_for(spec))

    assert "strict_constant" not in loaded.df.columns
    assert "strict_constant" not in set(loaded.registry.to_dataframe()["name"])
    assert "mostly_null" not in loaded.df.columns
    assert {"row_id", "target", "metadata_all_null", "usable", "SPLIT_PCT"}.issubset(
        loaded.df.columns
    )
    assert loaded.registry.get("metadata_all_null").model is False
    assert loaded.registry.get("row_id").model is False
    assert loaded.registry.get("target").target is True


def test_strict_constant_drop_preserves_constant_target_hash_and_metadata_columns(tmp_path):
    csv_path = tmp_path / "protected_constant.csv"
    pd.DataFrame(
        {
            "row_id": [1, 2, 3],
            "TARGET": [1, 1, 1],
            "metadata_constant": ["same", "same", "same"],
            "strict_constant": [7, 7, 7],
            "usable": [10, 20, 30],
        }
    ).to_csv(csv_path, index=False)
    spec = DataSpec(
        source=LocalCSVSource(csv_path=csv_path, unique_key="row_id"),
        metadata_cols=("metadata_constant",),
        constant_drop_threshold=1.0,
        dry_run_rows=10,
    )

    loaded = build_dataset(session=_session_for(spec))
    registry_names = set(loaded.registry.to_dataframe()["name"])

    assert "strict_constant" not in loaded.df.columns
    assert "strict_constant" not in registry_names
    assert {"row_id", "target", "metadata_constant"}.issubset(loaded.df.columns)
    assert {"row_id", "target", "metadata_constant"}.issubset(registry_names)


def test_build_dataset_errors_on_duplicate_unique_key(tmp_path):
    csv_path = tmp_path / "dups.csv"
    pd.DataFrame({"row_id": [1, 1, 2], "x": [0.1, 0.2, 0.3], "TARGET": [0, 1, 0]}).to_csv(
        csv_path, index=False
    )
    spec = DataSpec(source=LocalCSVSource(csv_path=csv_path, unique_key="row_id"), dry_run_rows=10)

    with pytest.raises(DataError, match="duplicate"):
        build_dataset(session=_session_for(spec))


def test_build_dataset_errors_when_source_provides_split_pct(tmp_path):
    csv_path = tmp_path / "collide.csv"
    pd.DataFrame({"row_id": [1, 2], "SPLIT_PCT": [3, 4], "TARGET": [0, 1]}).to_csv(
        csv_path, index=False
    )
    spec = DataSpec(source=LocalCSVSource(csv_path=csv_path, unique_key="row_id"), dry_run_rows=10)

    with pytest.raises(DataError, match="SPLIT_PCT"):
        build_dataset(session=_session_for(spec))


def test_build_dataset_groups_splits_by_split_group_key(tmp_path):
    csv_path = tmp_path / "grouped.csv"
    pd.DataFrame(
        {"txn_id": [1, 2, 3, 4], "user_id": ["a", "a", "b", "b"], "TARGET": [0, 1, 0, 1]}
    ).to_csv(csv_path, index=False)
    spec = DataSpec(
        source=LocalCSVSource(csv_path=csv_path, unique_key="txn_id", split_group_key="user_id"),
        dry_run_rows=10,
    )

    loaded = build_dataset(session=_session_for(spec))

    buckets = loaded.df.groupby("user_id")["SPLIT_PCT"].nunique()
    assert (buckets == 1).all()  # one user never straddles buckets


def test_trial_data_contract_round_trips_distinct_trial_id_and_run_id():
    contract = TrialDataContract(
        trial=TrialRef(
            project_name="demo",
            experiment_id="baseline",
            trial_id="001_first_model",
            run_id="b179a36c7ebc4f2fb612b4dcf1ad353c",
        ),
        dataset=DatasetRef(
            id="v1_abcdef12",
            manifest_uri="gs://bucket/demo/data/datasets/v1_abcdef12/manifest.json",
            identity_hash="sha256:identity",
            target_column="target",
            split_pct_col="SPLIT_PCT",
            n_rows=10,
            n_columns=4,
        ),
        splits={"train": ((0, 80),), "test": ((80, 100),)},
        slices=(
            SliceContract(
                name="train",
                ranges=((0, 80),),
                n_rows=8,
                content_hash="sha256:train",
            ),
        ),
    )

    restored = TrialDataContract.from_dict(contract.to_dict())

    assert restored.trial.trial_id == "001_first_model"
    assert restored.trial.run_id == "b179a36c7ebc4f2fb612b4dcf1ad353c"
    assert restored.trial.trial_id != restored.trial.run_id
    assert restored.slice("train").content_hash == "sha256:train"
    assert restored.slice("test") is None
