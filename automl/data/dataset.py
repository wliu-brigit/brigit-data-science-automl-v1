"""Dataset identity and loaded data value objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd

from automl.data.features import FeatureRegistry


@dataclass(frozen=True)
class ComponentHashes:
    source_identity: str
    feature_registry: str
    data_content: str
    schema: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ComponentHashes":
        return cls(
            source_identity=str(payload.get("source_identity", "")),
            feature_registry=str(payload.get("feature_registry", "")),
            data_content=str(payload.get("data_content", "")),
            schema=str(payload.get("schema", "")),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "source_identity": self.source_identity,
            "feature_registry": self.feature_registry,
            "data_content": self.data_content,
            "schema": self.schema,
        }


@dataclass(frozen=True)
class Dataset:
    id: str
    identity_hash: str
    component_hashes: ComponentHashes
    gcs_bucket: str
    project_name: str
    created_at: str
    source_identity: dict[str, Any]
    n_rows: int
    n_columns: int
    target_column: str
    split_pct_col: str
    unique_key: tuple[str, ...]
    split_group_key: tuple[str, ...]
    gcs_prefix: str = ""
    experiment_id: str = ""
    recipe: Mapping[str, Any] = field(default_factory=dict)  # readable dict, design §3
    record_uri: str = ""  # runs:/<overview_run>/datasets/<id>/dataset.json, set at persist
    schema_version: int = 1

    @property
    def gcs_base_path(self) -> str:
        parts = [
            part.strip("/")
            for part in (self.gcs_prefix, self.project_name, self.experiment_id)
            if part.strip("/")
        ]
        parts.extend(["data", "datasets", self.id])
        return "/".join(parts)

    @property
    def data_gcs_uri(self) -> str:
        return f"gs://{self.gcs_bucket}/{self.gcs_base_path}/data.parquet"

    @property
    def registry_gcs_uri(self) -> str:
        return f"gs://{self.gcs_bucket}/{self.gcs_base_path}/feature_registry.csv"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Dataset":
        hashes = payload.get("component_hashes", payload.get("hashes", {}))
        if not isinstance(hashes, Mapping):
            hashes = {}
        return cls(
            id=str(payload.get("id", payload.get("dataset_id", ""))),
            identity_hash=str(payload.get("identity_hash", "")),
            component_hashes=ComponentHashes.from_dict(hashes),
            gcs_bucket=str(payload.get("gcs_bucket", "")),
            project_name=str(payload.get("project_name", "")),
            created_at=str(payload.get("created_at", "")),
            source_identity=dict(payload.get("source_identity", {})),
            n_rows=int(payload.get("n_rows", 0)),
            n_columns=int(payload.get("n_columns", 0)),
            target_column=str(payload.get("target_column", "")),
            split_pct_col=str(payload.get("split_pct_col", "SPLIT_PCT")),
            unique_key=tuple(str(item) for item in payload.get("unique_key", ())),
            split_group_key=tuple(str(item) for item in payload.get("split_group_key", ())),
            gcs_prefix=str(payload.get("gcs_prefix", "")),
            experiment_id=str(payload.get("experiment_id", "")),
            recipe=dict(payload.get("recipe", {})),
            record_uri=str(payload.get("record_uri", "")),
            schema_version=int(payload.get("schema_version", 1)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "identity_hash": self.identity_hash,
            "component_hashes": self.component_hashes.to_dict(),
            "gcs_bucket": self.gcs_bucket,
            "gcs_prefix": self.gcs_prefix,
            "experiment_id": self.experiment_id,
            "project_name": self.project_name,
            "created_at": self.created_at,
            "source_identity": self.source_identity,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "target_column": self.target_column,
            "split_pct_col": self.split_pct_col,
            "unique_key": list(self.unique_key),
            "split_group_key": list(self.split_group_key),
            "recipe": dict(self.recipe),
            "data_gcs_uri": self.data_gcs_uri,
            "registry_gcs_uri": self.registry_gcs_uri,
            # record_uri deliberately absent: the persisted record never
            # stores a pointer to itself; read_dataset_record injects it.
        }


@dataclass(frozen=True)
class LoadedDataset:
    dataset: Dataset
    df: pd.DataFrame
    registry: FeatureRegistry

    @property
    def id(self) -> str:
        return self.dataset.id

    @property
    def n_rows(self) -> int:
        return len(self.df)


@dataclass(frozen=True)
class LoadedSlice:
    dataset: Dataset
    df: pd.DataFrame
    registry: FeatureRegistry
    split_name: str | None
    # A Predicate (automl.project.predicates); typed Any to keep the data
    # layer's import surface unchanged — the object arrives via registry.py.
    predicate: Any

    @property
    def id(self) -> str:
        return self.dataset.id

    @property
    def n_rows(self) -> int:
        return len(self.df)


@dataclass(frozen=True)
class DatasetIndex:
    """In-memory view assembled from the per-version records; never persisted."""

    datasets: tuple[Dataset, ...]
    active_dataset_id: str | None = None

    @property
    def active(self) -> Dataset | None:
        if self.active_dataset_id is None:
            return None
        for dataset in self.datasets:
            if dataset.id == self.active_dataset_id:
                return dataset
        return None

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([dataset.to_dict() for dataset in self.datasets])


__all__ = ["ComponentHashes", "Dataset", "DatasetIndex", "LoadedDataset", "LoadedSlice"]
