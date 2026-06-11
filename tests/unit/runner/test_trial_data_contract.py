from __future__ import annotations

import pandas as pd
import pytest

from automl.data.dataset import Dataset, LoadedDataset, LoadedSlice
from automl.data.features import FeatureRegistry
from automl.project import Where
from automl.project.run_config import ModelRoute, ModelsConfig, RunConfig, Splits
from automl.runner import trial as trial_module
from automl.utils.hashing import dataframe_content_hash

pytestmark = pytest.mark.unit


_FRAME = pd.DataFrame(
    {
        "SPLIT_PCT": [10, 30, 70, 85, 95],
        "y": [0, 1, 0, 1, 0],
    }
)


def _dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "id": "ds_001",
            "identity_hash": "sha256:identity",
            "component_hashes": {
                "source_identity": "sha256:src",
                "feature_registry": "sha256:reg",
                "data_content": "sha256:content",
                "schema": "sha256:schema",
            },
            "gcs_bucket": "bucket-x",
            "project_name": "proj",
            "created_at": "2026-06-10T00:00:00Z",
            "source_identity": {},
            "n_rows": 5,
            "n_columns": 2,
            "target_column": "y",
        }
    )


def _registry() -> FeatureRegistry:
    return FeatureRegistry.from_dataframe(
        pd.DataFrame({"name": ["y"], "dtype": ["int64"]})
    )


def _run_config() -> RunConfig:
    route = ModelRoute(model="claude-test", effort="low")
    return RunConfig(
        experiment_id="exp",
        splits=Splits(
            train=Where("SPLIT_PCT") < 80,
            test=Where("SPLIT_PCT") >= 80,
        ),
        models=ModelsConfig(manager=route, proposer=route, coder=route),
        per_trial_seconds=600,
    )


class _Config:
    def require_run_config(self):
        return _run_config()


class _Session:
    config = _Config()
    project_name = "proj"
    active_experiment_id = "1"


def _loaded_fit() -> LoadedSlice:
    predicate = _run_config().splits.resolve("train")
    df = _FRAME[predicate.mask(_FRAME)].reset_index(drop=True)
    return LoadedSlice(
        dataset=_dataset(),
        df=df,
        registry=_registry(),
        split_name="train",
        predicate=predicate,
    )


def test_contract_loads_full_frame_exactly_once(monkeypatch):
    calls = []

    def fake_load(dataset_id, *, split_name=None, predicate=None, session=None):
        calls.append({"dataset_id": dataset_id, "split_name": split_name})
        assert split_name is None and predicate is None, "must load the FULL frame"
        return LoadedDataset(dataset=_dataset(), df=_FRAME.copy(), registry=_registry())

    monkeypatch.setattr(trial_module.data, "load_dataset_by_id", fake_load)
    contract = trial_module._trial_data_contract(
        active=_Session(),
        run_id="run1",
        trial_id="1_test",
        loaded_fit=_loaded_fit(),
    )
    assert len(calls) == 1
    assert calls[0]["dataset_id"] == "ds_001"
    assert {s.name for s in contract.slices} == {"train", "test"}


def test_slice_hashes_match_direct_slicing(monkeypatch):
    monkeypatch.setattr(
        trial_module.data,
        "load_dataset_by_id",
        lambda *a, **k: LoadedDataset(
            dataset=_dataset(), df=_FRAME.copy(), registry=_registry()
        ),
    )
    contract = trial_module._trial_data_contract(
        active=_Session(),
        run_id="run1",
        trial_id="1_test",
        loaded_fit=_loaded_fit(),
    )
    run_config = _run_config()
    assert len(contract.slices) == 2
    for slice_contract in contract.slices:
        predicate = run_config.splits.resolve(slice_contract.name)
        expected = _FRAME[predicate.mask(_FRAME)].reset_index(drop=True)
        assert slice_contract.n_rows == len(expected)
        assert slice_contract.content_hash == dataframe_content_hash(expected)
        assert slice_contract.predicate == predicate.to_dict()
