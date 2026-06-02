"""Trial data-contract types and integrity validators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from automl.data.dataset import Dataset, LoadedDataset, LoadedSlice
from automl.errors import DataError
from automl.utils.hashing import dataframe_content_hash, schema_hash


@dataclass(frozen=True)
class TrialRef:
    project_name: str
    experiment_id: str
    trial_id: str
    run_id: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrialRef":
        return cls(
            project_name=str(payload.get("project_name", "")),
            experiment_id=str(payload.get("experiment_id", "")),
            trial_id=str(payload.get("trial_id", "")),
            run_id=str(payload.get("run_id", "")),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "project_name": self.project_name,
            "experiment_id": self.experiment_id,
            "trial_id": self.trial_id,
            "run_id": self.run_id,
        }


@dataclass(frozen=True)
class DatasetRef:
    id: str
    manifest_uri: str
    identity_hash: str
    target_column: str
    split_id_col: str
    n_rows: int
    n_columns: int

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> "DatasetRef":
        return cls(
            id=dataset.id,
            manifest_uri=dataset.manifest_gcs_uri,
            identity_hash=dataset.identity_hash,
            target_column=dataset.target_column,
            split_id_col=dataset.split_id_col,
            n_rows=dataset.n_rows,
            n_columns=dataset.n_columns,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DatasetRef":
        return cls(
            id=str(payload.get("id", "")),
            manifest_uri=str(payload.get("manifest_uri", "")),
            identity_hash=str(payload.get("identity_hash", "")),
            target_column=str(payload.get("target_column", "")),
            split_id_col=str(payload.get("split_id_col", "")),
            n_rows=int(payload.get("n_rows", 0)),
            n_columns=int(payload.get("n_columns", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class SliceContract:
    name: str | None
    ranges: tuple[tuple[int, int], ...]
    n_rows: int
    content_hash: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SliceContract":
        return cls(
            name=payload.get("name"),
            ranges=_ranges_from(payload.get("ranges", ())),
            n_rows=int(payload.get("n_rows", 0)),
            content_hash=str(payload.get("content_hash", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ranges": [[low, high] for low, high in self.ranges],
            "n_rows": self.n_rows,
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True)
class TrialDataContract:
    trial: TrialRef
    dataset: DatasetRef
    splits: dict[str, tuple[tuple[int, int], ...]]
    slices: tuple[SliceContract, ...]
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trial": self.trial.to_dict(),
            "dataset": self.dataset.to_dict(),
            "splits": {
                name: [[low, high] for low, high in ranges]
                for name, ranges in self.splits.items()
            },
            "slices": [slice_contract.to_dict() for slice_contract in self.slices],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrialDataContract":
        return cls(
            trial=TrialRef.from_dict(_mapping(payload.get("trial"))),
            dataset=DatasetRef.from_dict(_mapping(payload.get("dataset"))),
            splits={
                str(name): _ranges_from(ranges)
                for name, ranges in _mapping(payload.get("splits")).items()
            },
            slices=tuple(
                SliceContract.from_dict(_mapping(item)) for item in payload.get("slices", ())
            ),
            schema_version=int(payload.get("schema_version", 1)),
        )

    def slice(self, name: str) -> SliceContract | None:
        for slice_contract in self.slices:
            if slice_contract.name == name:
                return slice_contract
        return None


def validate_trial_data_contract(contract: TrialDataContract, dataset: Dataset) -> None:
    checks = {
        "id": (contract.dataset.id, dataset.id),
        "identity_hash": (contract.dataset.identity_hash, dataset.identity_hash),
        "target_column": (contract.dataset.target_column, dataset.target_column),
        "split_id_col": (contract.dataset.split_id_col, dataset.split_id_col),
        "n_rows": (contract.dataset.n_rows, dataset.n_rows),
        "n_columns": (contract.dataset.n_columns, dataset.n_columns),
    }
    for field, (left, right) in checks.items():
        if left != right:
            raise DataError(f"trial data contract {field} does not match dataset")


def validate_loaded_dataset(loaded: LoadedDataset, dataset: Dataset) -> None:
    if len(loaded.df) != dataset.n_rows:
        raise DataError("loaded dataset n_rows does not match manifest")
    if len(loaded.df.columns) != dataset.n_columns:
        raise DataError("loaded dataset n_columns does not match manifest")
    expected = dataset.component_hashes
    actual = {
        "data_content": dataframe_content_hash(loaded.df),
        "feature_registry": loaded.registry.content_hash(),
        "schema": schema_hash(loaded.df),
    }
    for field, value in actual.items():
        if getattr(expected, field) != value:
            raise DataError(f"loaded dataset {field} hash does not match manifest")


def verify_loaded_slice(loaded: LoadedSlice, slice_contract: SliceContract) -> None:
    if loaded.n_rows != slice_contract.n_rows:
        raise DataError("loaded slice n_rows does not match contract")
    if dataframe_content_hash(loaded.df) != slice_contract.content_hash:
        raise DataError("loaded slice content_hash does not match contract")


def verify_trial_tag_lineage(contract: TrialDataContract, run_id: str) -> None:
    from automl.mlflow import trial as mlflow_trial

    run_tags = mlflow_trial.get_tags(run_id)
    expected = {
        "data.dataset_id": contract.dataset.id,
        "data.identity_hash": contract.dataset.identity_hash,
        "data.manifest_uri": contract.dataset.manifest_uri,
    }
    for slice_contract in contract.slices:
        if slice_contract.name is None:
            continue
        expected[f"data.slice.{slice_contract.name}.content_hash"] = (
            slice_contract.content_hash
        )
    for key, value in expected.items():
        actual = run_tags.get(key)
        if actual != value:
            raise DataError(
                f"trial tag lineage mismatch for {key}: expected {value!r}, got {actual!r}"
            )


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"expected mapping, got {type(value).__name__}")
    return value


def _ranges_from(value: object) -> tuple[tuple[int, int], ...]:
    return tuple((int(low), int(high)) for low, high in value)


__all__ = [
    "DatasetRef",
    "SliceContract",
    "TrialDataContract",
    "TrialRef",
    "validate_loaded_dataset",
    "validate_trial_data_contract",
    "verify_loaded_slice",
    "verify_trial_tag_lineage",
]
