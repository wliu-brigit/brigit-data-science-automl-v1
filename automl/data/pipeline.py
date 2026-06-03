"""Thin data pipeline and materialization entry points."""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from automl.data.dataset import ComponentHashes, Dataset, DatasetIndex, LoadedDataset
from automl.data.features import FeatureRegistry
from automl.data.split import ROW_FALLBACK_HASH_KEY, add_split_id, hash_key_columns
from automl.errors import DataError, ProjectError
from automl.mlflow import client as mlflow_client
from automl.mlflow import experiment as mlflow_experiment
from automl.mlflow import routing as mlflow_routing
from automl.mlflow.experiment import artifacts as experiment_artifacts
from automl.project import Session
from automl.project import session as active_project_session
from automl.utils.hashing import dataframe_content_hash, json_hash, schema_hash
from automl.utils.io import gcs


class DataPipeline:
    split_id_col = "SPLITID"

    def __init__(self, spec, session: Session, *, refresh_source: bool = False) -> None:
        self.spec = spec
        self.session = session
        self.refresh_source = refresh_source

    def run(self) -> LoadedDataset:
        raw = self.spec.source.load(
            project_dir=self.session.config.project_dir,
            nrows=self.spec.dry_run_rows if self.session.dry_run else None,
        )
        df, original_names = self.standardize_columns(raw)
        hash_key = self._normalized_hash_key(original_names)
        target_column = self._normalized_target(original_names, df)
        metadata_cols = self._normalize_declared(self.spec.metadata_cols, original_names)
        registry_metadata_cols = _unique_tuple((*metadata_cols, *hash_key))
        exclude_cols = self._normalize_declared(self.spec.exclude_cols, original_names)
        df = self._apply_quality_filters(
            df,
            protected_cols=(target_column, *hash_key, *metadata_cols),
        )
        df = add_split_id(df, hash_key=None if hash_key == (ROW_FALLBACK_HASH_KEY,) else hash_key)
        registry = FeatureRegistry().build_from_df(
            df,
            target_column=target_column,
            metadata_cols=registry_metadata_cols,
            exclude_cols=exclude_cols,
            split_id_col=self.split_id_col,
            original_names=original_names,
        )
        dataset = self._dataset_for(
            df,
            registry,
            hash_key=hash_key,
            target_column=target_column,
        )
        return LoadedDataset(dataset=dataset, df=df, registry=registry)

    def standardize_columns(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
        normalized: list[str] = []
        emitted: set[str] = set()
        for column in df.columns:
            base = _normalize_column(str(column))
            candidate = base
            suffix = 2
            while candidate in emitted:
                candidate = f"{base}_{suffix}"
                suffix += 1
            emitted.add(candidate)
            normalized.append(candidate)
        original_names = dict(zip(normalized, [str(column) for column in df.columns], strict=True))
        out = df.copy()
        out.columns = normalized
        return out, original_names

    def _normalized_hash_key(self, original_names: dict[str, str]) -> tuple[str, ...]:
        source_hash_key = getattr(self.spec.source, "hash_key", None)
        if source_hash_key is None:
            return (ROW_FALLBACK_HASH_KEY,)
        raw_to_normalized = {raw: normalized for normalized, raw in original_names.items()}
        return tuple(
            raw_to_normalized.get(column, _normalize_column(column))
            for column in hash_key_columns(source_hash_key)
        )

    def _normalize_declared(
        self, values: tuple[str, ...], original_names: dict[str, str]
    ) -> tuple[str, ...]:
        raw_to_normalized = {raw: normalized for normalized, raw in original_names.items()}
        return tuple(raw_to_normalized.get(value, _normalize_column(value)) for value in values)

    def _normalized_target(self, original_names: dict[str, str], df: pd.DataFrame) -> str:
        raw_to_normalized = {raw: normalized for normalized, raw in original_names.items()}
        target_column = raw_to_normalized.get(
            self.session.config.raw_target_column,
            self.session.config.target_column,
        )
        if target_column not in df.columns:
            raise DataError(
                f"target column {self.session.config.raw_target_column!r} "
                f"not found after standardization"
            )
        return target_column

    def _apply_quality_filters(
        self,
        df: pd.DataFrame,
        *,
        protected_cols: tuple[str, ...],
    ) -> pd.DataFrame:
        protected = set(protected_cols)
        keep: list[str] = []
        for column in df.columns:
            if column in protected:
                keep.append(column)
                continue
            series = df[column]
            null_pct = float(series.isna().mean()) if len(series) else 0.0
            if null_pct > float(self.spec.null_drop_threshold):
                continue
            dominance_pct = _dominance_pct(series)
            if dominance_pct >= float(self.spec.constant_drop_threshold):
                continue
            keep.append(column)
        return df.loc[:, keep]

    def _dataset_for(
        self,
        df: pd.DataFrame,
        registry: FeatureRegistry,
        *,
        hash_key: tuple[str, ...],
        target_column: str,
        dataset_id: str = "unmaterialized",
    ) -> Dataset:
        source_identity = dict(self.spec.source.identity())
        source_identity["hash_key"] = list(hash_key)
        component_hashes = ComponentHashes(
            source_identity=json_hash(source_identity),
            feature_registry=registry.content_hash(),
            data_content=dataframe_content_hash(df),
            schema=schema_hash(df),
        )
        identity_hash = json_hash(
            {
                "component_hashes": component_hashes.to_dict(),
                "hash_key": list(hash_key),
                "n_columns": len(df.columns),
                "n_rows": len(df),
                "project_name": self.session.project_name,
                "split_id_col": self.split_id_col,
                "target_column": target_column,
            }
        )
        return Dataset(
            id=dataset_id,
            identity_hash=identity_hash,
            component_hashes=component_hashes,
            gcs_bucket=self.session.config.gcs_bucket,
            gcs_prefix=_dataset_gcs_prefix(self.session),
            experiment_id=_dataset_experiment_id(self.session),
            project_name=self.session.project_name,
            created_at=datetime.now(UTC).isoformat(),
            source_identity=source_identity,
            n_rows=len(df),
            n_columns=len(df.columns),
            target_column=target_column,
            split_id_col=self.split_id_col,
            hash_key=hash_key,
        )


def build_dataset(*, session: Session | None = None) -> LoadedDataset:
    active = _session(session)
    spec = active.config.require_data_spec()
    return spec.pipeline_cls(spec, active).run()


def materialize(
    *,
    refresh_source: bool = False,
    include_rows: bool = True,
    session: Session | None = None,
) -> LoadedDataset | Dataset:
    """Materialize data; include_rows controls return shape only, not persistence."""
    active = _session(session)
    with mlflow_client.bound_for(active, experiment_id=active.active_experiment_id):
        loaded = _materialize_bound(active=active, refresh_source=refresh_source)
    return loaded if include_rows else loaded.dataset


def _materialize_bound(*, active: Session, refresh_source: bool) -> LoadedDataset:
    spec = active.config.require_data_spec()
    pipeline = spec.pipeline_cls(spec, active, refresh_source=refresh_source)
    loaded = pipeline.run()
    index = DatasetIndex.from_dict(experiment_artifacts.read_dataset_index())
    existing = _dataset_for_identity(index, loaded.dataset.identity_hash)
    dataset_id = existing.id if existing is not None else _next_dataset_id(index, loaded.dataset)
    dataset = replace(
        loaded.dataset,
        id=dataset_id,
        created_at=existing.created_at if existing else loaded.dataset.created_at,
    )
    loaded = LoadedDataset(dataset=dataset, df=loaded.df, registry=loaded.registry)

    object_state = _dataset_object_state(dataset)
    if existing is not None and all(object_state.values()):
        manifest = experiment_artifacts.read_dataset_manifest(dataset.manifest_gcs_uri)
        persisted = Dataset.from_dict(manifest)
        _validate_existing_dataset_matches_candidate(persisted, dataset)
        mlflow_experiment.set_active_dataset(dataset.id, experiment_id=active.active_experiment_id)
        experiment_artifacts.log_dataset_catalog(
            index.to_dict(),
            active_dataset_id=dataset.id,
        )
        return LoadedDataset(dataset=persisted, df=loaded.df, registry=loaded.registry)
    if any(object_state.values()) and not all(object_state.values()):
        present = [name for name, exists in object_state.items() if exists]
        missing = [name for name, exists in object_state.items() if not exists]
        raise DataError(
            f"partial dataset objects for {dataset.id}: present={present} missing={missing}"
        )

    experiment_artifacts.write_dataset_frame(dataset.data_gcs_uri, loaded.df)
    experiment_artifacts.write_registry(dataset.registry_gcs_uri, loaded.registry.to_dataframe())
    experiment_artifacts.write_dataset_manifest(dataset.manifest_gcs_uri, dataset.to_dict())
    _log_source_trace(dataset, spec.source.artifact_files(pipeline))

    datasets = tuple(item for item in index.datasets if item.id != dataset.id) + (dataset,)
    next_index = DatasetIndex(datasets=datasets)
    experiment_artifacts.write_dataset_index(next_index.to_dict())
    mlflow_experiment.set_active_dataset(dataset.id, experiment_id=active.active_experiment_id)
    experiment_artifacts.log_dataset_catalog(
        next_index.to_dict(),
        active_dataset_id=dataset.id,
    )
    return loaded


def _session(explicit: Session | None) -> Session:
    return explicit if explicit is not None else active_project_session()


def _dataset_gcs_prefix(active: Session) -> str:
    return mlflow_routing.namespace_route_prefix_for(
        gcs_prefix=active.config.gcs_prefix,
        dry_run=active.dry_run,
        namespace=active.namespace,
    )


def _dataset_experiment_id(active: Session) -> str:
    """Datasets are experiment-owned; exploration sessions build unmaterialized ones."""
    try:
        return active.active_experiment_id
    except ProjectError:
        return ""


def _log_source_trace(dataset: Dataset, source_files) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        identity_path = Path(tmp_dir) / "source_identity.json"
        identity_path.write_text(
            json.dumps(dataset.source_identity, indent=2, default=str),
            encoding="utf-8",
        )
        files = {"source_identity.json": identity_path, **dict(source_files)}
        experiment_artifacts.log_source_trace(dataset.id, files)


def _dataset_for_identity(index: DatasetIndex, identity_hash: str) -> Dataset | None:
    for dataset in index.datasets:
        if dataset.identity_hash == identity_hash:
            return dataset
    return None


def _next_dataset_id(index: DatasetIndex, dataset: Dataset) -> str:
    max_version = 0
    for current in index.datasets:
        match = re.match(r"^v(\d+)_", current.id)
        if match:
            max_version = max(max_version, int(match.group(1)))
    return f"v{max_version + 1}_{dataset.identity_hash.removeprefix('sha256:')[:8]}"


def _dataset_object_state(dataset: Dataset) -> dict[str, bool]:
    return {
        "data": gcs.blob_exists(dataset.data_gcs_uri),
        "registry": gcs.blob_exists(dataset.registry_gcs_uri),
        "manifest": gcs.blob_exists(dataset.manifest_gcs_uri),
    }


def _validate_existing_dataset_matches_candidate(existing: Dataset, candidate: Dataset) -> None:
    mismatches: list[str] = []
    for field in (
        "id",
        "identity_hash",
        "component_hashes",
        "source_identity",
        "n_rows",
        "n_columns",
        "target_column",
        "split_id_col",
        "hash_key",
    ):
        if getattr(existing, field) != getattr(candidate, field):
            mismatches.append(field)
    if mismatches:
        raise DataError(
            f"existing dataset {candidate.id} does not match candidate fields: {mismatches}"
        )


def _normalize_column(name: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z_]+", "_", name.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        raise DataError(f"column name {name!r} normalizes to empty string")
    return normalized


def _dominance_pct(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    counts = series.value_counts(dropna=False)
    if counts.empty:
        return 0.0
    return float(counts.iloc[0] / len(series))


def _unique_tuple(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


__all__ = ["DataPipeline", "build_dataset", "materialize"]
