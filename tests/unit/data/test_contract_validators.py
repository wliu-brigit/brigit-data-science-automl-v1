from dataclasses import replace

import pandas as pd
import pytest

from automl.data import (
    ComponentHashes,
    Dataset,
    DatasetRef,
    FeatureRegistry,
    LoadedDataset,
    LoadedSlice,
    SliceContract,
    TrialDataContract,
    TrialRef,
    validate_loaded_dataset,
    validate_trial_data_contract,
    verify_loaded_slice,
    verify_trial_tag_lineage,
)
from automl.errors import DataError
from automl.mlflow import trial as mlflow_trial
from automl.utils.hashing import dataframe_content_hash, schema_hash

pytestmark = pytest.mark.unit


def _loaded_dataset() -> LoadedDataset:
    df = pd.DataFrame(
        {
            "row_id": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
            "amount": [10.0, 20.0, 30.0, 40.0],
            "SPLIT_PCT": [10, 20, 90, 95],
        }
    )
    registry = FeatureRegistry().build_from_df(
        df,
        target_column="target",
        metadata_cols=("row_id",),
        split_pct_col="SPLIT_PCT",
    )
    dataset = Dataset(
        id="v1_abc12345",
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
        split_pct_col="SPLIT_PCT",
        unique_key=("row_id",),
        split_group_key=("row_id",),
    )
    return LoadedDataset(dataset=dataset, df=df, registry=registry)


def _contract(loaded: LoadedDataset) -> TrialDataContract:
    train_df = loaded.df[loaded.df["SPLIT_PCT"].isin(range(0, 50))].reset_index(drop=True)
    return TrialDataContract(
        trial=TrialRef(
            project_name="demo",
            experiment_id="baseline",
            trial_id="1_replay",
            run_id="run-123",
        ),
        dataset=DatasetRef.from_dataset(loaded.dataset),
        splits={"train": ((0, 50),), "holdout": ((90, 100),)},
        slices=(
            SliceContract(
                name="train",
                ranges=((0, 50),),
                n_rows=len(train_df),
                content_hash=dataframe_content_hash(train_df),
            ),
        ),
    )


@pytest.mark.parametrize(
    ("field", "candidate"),
    [
        ("id", lambda dataset: replace(dataset, id="different")),
        ("identity_hash", lambda dataset: replace(dataset, identity_hash="sha256:different")),
        ("target_column", lambda dataset: replace(dataset, target_column="label")),
        ("split_pct_col", lambda dataset: replace(dataset, split_pct_col="split")),
        ("n_rows", lambda dataset: replace(dataset, n_rows=dataset.n_rows + 1)),
        ("n_columns", lambda dataset: replace(dataset, n_columns=dataset.n_columns + 1)),
    ],
)
def test_l1_trial_data_contract_rejects_dataset_mismatch(field, candidate):
    loaded = _loaded_dataset()
    contract = _contract(loaded)

    with pytest.raises(DataError, match=field):
        validate_trial_data_contract(contract, candidate(loaded.dataset))


@pytest.mark.parametrize(
    ("hash_field", "expected_message"),
    [
        ("data_content", "data_content"),
        ("feature_registry", "feature_registry"),
        ("schema", "schema"),
    ],
)
def test_l2_loaded_dataset_rejects_component_hash_drift(hash_field, expected_message):
    loaded = _loaded_dataset()
    hashes = replace(loaded.dataset.component_hashes, **{hash_field: "sha256:corrupt"})
    dataset = replace(loaded.dataset, component_hashes=hashes)

    with pytest.raises(DataError, match=expected_message):
        validate_loaded_dataset(loaded, dataset)


def test_l3_loaded_slice_rejects_row_count_and_content_hash_drift():
    loaded = _loaded_dataset()
    loaded_slice = LoadedSlice(
        dataset=loaded.dataset,
        df=loaded.df.iloc[:2].reset_index(drop=True),
        registry=loaded.registry,
        split_name="train",
        split_ranges=((0, 50),),
    )
    contract = SliceContract(
        name="train",
        ranges=((0, 50),),
        n_rows=2,
        content_hash=dataframe_content_hash(loaded_slice.df),
    )

    with pytest.raises(DataError, match="n_rows"):
        verify_loaded_slice(loaded_slice, replace(contract, n_rows=3))
    with pytest.raises(DataError, match="content_hash"):
        verify_loaded_slice(loaded_slice, replace(contract, content_hash="sha256:corrupt"))


def test_l4_trial_tag_lineage_verifies_dataset_and_slice_tags(monkeypatch):
    loaded = _loaded_dataset()
    contract = _contract(loaded)
    good_tags = {
        "data.dataset_id": contract.dataset.id,
        "data.identity_hash": contract.dataset.identity_hash,
        "data.record_uri": contract.dataset.record_uri,
        "data.slice.train.content_hash": contract.slice("train").content_hash,
    }
    monkeypatch.setattr(mlflow_trial, "get_tags", lambda run_id: good_tags, raising=False)

    verify_trial_tag_lineage(contract, "run-123")

    for key in good_tags:
        bad_tags = dict(good_tags)
        bad_tags[key] = "corrupt"
        monkeypatch.setattr(
            mlflow_trial, "get_tags", lambda run_id, tags=bad_tags: tags, raising=False
        )
        with pytest.raises(DataError, match=key):
            verify_trial_tag_lineage(contract, "run-123")

        missing_tags = dict(good_tags)
        missing_tags.pop(key)
        monkeypatch.setattr(
            mlflow_trial,
            "get_tags",
            lambda run_id, tags=missing_tags: tags,
            raising=False,
        )
        with pytest.raises(DataError, match=key):
            verify_trial_tag_lineage(contract, "run-123")
