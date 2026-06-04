from pathlib import Path

import pandas as pd
import pytest

from automl.data import (
    DatasetRef,
    DataSpec,
    LocalCSVSource,
    SliceContract,
    TrialDataContract,
    TrialRef,
    load_dataset_by_id,
    load_dataset_by_trial,
    materialize,
)
from automl.errors import DataError
from automl.mlflow import client as mlflow_client
from automl.mlflow import experiment as mlflow_experiment
from automl.mlflow import tags as mlflow_tags
from automl.mlflow import trial as mlflow_trial
from automl.mlflow.trial import artifacts
from automl.project import (
    BinaryClassification,
    ModelRoute,
    ModelsConfig,
    ProjectConfig,
    RunConfig,
    Session,
    Splits,
)
from automl.utils.hashing import dataframe_content_hash
from automl.utils.io import gcs

pytestmark = pytest.mark.integration


class FakeBlob:
    def __init__(self, store: dict[tuple[str, str], bytes], bucket: str, name: str) -> None:
        self._store = store
        self._bucket = bucket
        self.name = name

    def upload_from_string(
        self,
        data: str | bytes,
        *,
        content_type: str | None = None,
        if_generation_match: int | None = None,
    ) -> None:
        if if_generation_match == 0 and (self._bucket, self.name) in self._store:
            raise FileExistsError(f"object already exists: {self._bucket}/{self.name}")
        self._store[(self._bucket, self.name)] = (
            data if isinstance(data, bytes) else data.encode("utf-8")
        )

    def upload_from_file(
        self,
        file_obj,
        *,
        content_type: str | None = None,
        if_generation_match: int | None = None,
    ) -> None:
        if if_generation_match == 0 and (self._bucket, self.name) in self._store:
            raise FileExistsError(f"object already exists: {self._bucket}/{self.name}")
        self._store[(self._bucket, self.name)] = file_obj.read()

    def download_as_bytes(self) -> bytes:
        return self._store[(self._bucket, self.name)]

    def exists(self) -> bool:
        return (self._bucket, self.name) in self._store


class FakeBucket:
    def __init__(self, store: dict[tuple[str, str], bytes], name: str) -> None:
        self._store = store
        self.name = name

    def blob(self, name: str) -> FakeBlob:
        return FakeBlob(self._store, self.name, name)


class FakeGCSClient:
    def __init__(self) -> None:
        self.store: dict[tuple[str, str], bytes] = {}

    def bucket(self, name: str) -> FakeBucket:
        return FakeBucket(self.store, name)


@pytest.fixture
def fake_gcs(monkeypatch) -> FakeGCSClient:
    fake = FakeGCSClient()
    monkeypatch.setattr(gcs, "_gcs_client", lambda: fake)
    yield fake
    mlflow_client.clear()


def _models() -> ModelsConfig:
    route = ModelRoute("sonnet", "medium")
    return ModelsConfig(manager=route, proposer=route, coder=route)


def _session(tmp_path: Path, csv_path: Path) -> Session:
    spec = DataSpec(source=LocalCSVSource(csv_path=csv_path, hash_key="row_id"))
    return Session(
        config=ProjectConfig(
            project_name="demo",
            repo_root=tmp_path,
            project_dir=tmp_path / "projects" / "demo",
            config_path=tmp_path / "projects" / "demo" / "config.py",
            task=BinaryClassification(target="target"),
            data_spec=spec,
            run_config=RunConfig(
                experiment_id="baseline",
                splits=Splits({"train": ((0, 50),), "test": ((50, 100),)}),
                models=_models(),
                per_trial_seconds=120,
            ),
            gcs_bucket="automl-test-bucket",
            gcs_prefix="automl-root",
            mlflow_tracking_uri=(tmp_path / "mlruns").as_uri(),
        )
    )


def _write_csv(tmp_path: Path) -> Path:
    csv_path = tmp_path / "tiny.csv"
    pd.DataFrame(
        {
            "row_id": list(range(1, 13)),
            "target": [0, 1] * 6,
            "amount": [100, 80, 95, 130, 50, 120, 99, 101, 75, 60, 42, 88],
        }
    ).to_csv(csv_path, index=False)
    return csv_path


def _write_trial_contract(active: Session, loaded, *, corrupt_tag: bool = False) -> str:
    with mlflow_client.bound_for(active, experiment_id=active.active_experiment_id):
        mlflow_experiment.ensure(experiment_id=active.active_experiment_id)
        trial_id = "1_replay_contract"
        splits = {"train": ((0, 50),), "holdout": ((90, 100),)}
        train = load_dataset_by_id(loaded.id, split_range=splits["train"], session=active)
        with mlflow_trial.active(
            slug="replay_contract",
            strategy="test",
            experiment_id=active.active_experiment_id,
        ) as run_id:
            mlflow_trial.set_tag(run_id, mlflow_tags.TRIAL_ID, trial_id)
            contract = TrialDataContract(
                trial=TrialRef(
                    project_name=active.project_name,
                    experiment_id=active.active_experiment_id,
                    trial_id=trial_id,
                    run_id=run_id,
                ),
                dataset=DatasetRef.from_dataset(loaded.dataset),
                splits=splits,
                slices=(
                    SliceContract(
                        name="train",
                        ranges=splits["train"],
                        n_rows=train.n_rows,
                        content_hash=dataframe_content_hash(train.df),
                    ),
                ),
            )
            artifacts.write_trial_data_contract(run_id, contract)
            mlflow_trial.set_tags(
                run_id,
                {
                    "data.dataset_id": contract.dataset.id,
                    "data.identity_hash": "sha256:corrupt"
                    if corrupt_tag
                    else contract.dataset.identity_hash,
                    "data.manifest_uri": contract.dataset.manifest_uri,
                    "data.slice.train.content_hash": contract.slice("train").content_hash,
                },
            )
        return trial_id


def test_load_dataset_by_id_accepts_disjoint_multi_range(tmp_path, fake_gcs):
    active = _session(tmp_path, _write_csv(tmp_path))
    loaded = materialize(session=active)

    sliced = load_dataset_by_id(
        loaded.id,
        split_range=((80, 90), (95, 100)),
        session=active,
    )
    expected_buckets = set(range(80, 90)) | set(range(95, 100))

    assert sliced.split_name is None
    assert sliced.split_ranges == ((80, 90), (95, 100))
    assert set(sliced.df["SPLIT_PCT"]).issubset(expected_buckets)
    assert sliced.df.to_dict("records") == loaded.df[
        loaded.df["SPLIT_PCT"].isin(expected_buckets)
    ].reset_index(drop=True).to_dict("records")


def test_load_dataset_by_trial_uses_contract_splits_and_runs_l3_l4(tmp_path, fake_gcs):
    active = _session(tmp_path, _write_csv(tmp_path))
    loaded = materialize(session=active)
    trial_id = _write_trial_contract(active, loaded)

    holdout = load_dataset_by_trial(trial_id, split_name="holdout", session=active)
    train = load_dataset_by_trial(trial_id, split_name="train", session=active)

    assert holdout.id == loaded.id
    assert holdout.split_name == "holdout"
    assert holdout.split_ranges == ((90, 100),)
    assert train.split_ranges == ((0, 50),)

    with pytest.raises(KeyError, match="available contract splits"):
        load_dataset_by_trial(trial_id, split_name="missing", session=active)


def test_load_dataset_by_trial_rejects_mismatched_l4_tags(tmp_path, fake_gcs):
    active = _session(tmp_path, _write_csv(tmp_path))
    loaded = materialize(session=active)
    trial_id = _write_trial_contract(active, loaded, corrupt_tag=True)

    with pytest.raises(DataError, match="data.identity_hash"):
        load_dataset_by_trial(trial_id, split_name="train", session=active)


def test_load_dataset_by_trial_can_skip_slice_contract_when_requested(tmp_path, fake_gcs):
    active = _session(tmp_path, _write_csv(tmp_path))
    loaded = materialize(session=active)
    trial_id = _write_trial_contract(active, loaded, corrupt_tag=True)

    with pytest.raises(DataError, match="data.identity_hash"):
        load_dataset_by_trial(trial_id, split_name="train", session=active)

    train = load_dataset_by_trial(
        trial_id,
        split_name="train",
        session=active,
        strict=False,
    )

    assert train.split_name == "train"
