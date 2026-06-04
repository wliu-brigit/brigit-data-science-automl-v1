"""Thin data pipeline and materialization entry points."""

from __future__ import annotations

import json
import logging
import re
import tempfile
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from automl.data.dataset import ComponentHashes, Dataset, LoadedDataset
from automl.data.features import FeatureRegistry
from automl.data.recipe import compute_recipe, recipe_diff
from automl.data.split import add_split_pct, validate_split_pct, validate_unique_key
from automl.errors import DataError, ProjectError
from automl.mlflow import client as mlflow_client
from automl.mlflow import experiment as mlflow_experiment
from automl.mlflow import routing as mlflow_routing
from automl.mlflow.experiment import artifacts as experiment_artifacts
from automl.project import Session
from automl.project import session as active_project_session
from automl.utils.hashing import dataframe_content_hash, json_hash, schema_hash
from automl.utils.io import gcs

logger = logging.getLogger(__name__)


class DataPipeline:
    split_pct_col = "SPLIT_PCT"

    def __init__(self, spec, session: Session, *, refresh_source: bool = False) -> None:
        self.spec = spec
        self.session = session
        self.refresh_source = refresh_source

    def run(self) -> LoadedDataset:
        raw = self.spec.source.load(
            project_dir=self.session.config.project_dir,
            nrows=self.spec.dry_run_rows if self.session.dry_run else None,
            refresh_source=self.refresh_source,
        )
        df, original_names = self.standardize_columns(raw)
        self._check_split_pct_collision(original_names)
        unique_key = self._normalize_key_columns(self.spec.source.unique_key_columns, original_names)
        split_group_key = self._normalize_key_columns(
            self.spec.source.split_group_key_columns, original_names
        )
        target_column = self._normalized_target(original_names, df)
        metadata_cols = self._normalize_declared(self.spec.metadata_cols, original_names)
        registry_metadata_cols = _unique_tuple((*metadata_cols, *unique_key, *split_group_key))
        exclude_cols = self._normalize_declared(self.spec.exclude_cols, original_names)
        df = self._apply_quality_filters(
            df,
            protected_cols=(target_column, *unique_key, *split_group_key, *metadata_cols),
        )
        if df.empty:
            raise DataError(
                f"materialized frame has 0 rows from {self.spec.source.kind}; an empty "
                "dataset is never useful — check the source data and quality thresholds"
            )
        df = add_split_pct(
            df, split_group_key=split_group_key, split_pct_col=self.split_pct_col
        )
        validate_unique_key(df, unique_key=unique_key, source_label=self.spec.source.kind)
        validate_split_pct(df, split_pct_col=self.split_pct_col, source_label=self.spec.source.kind)
        registry = FeatureRegistry().build_from_df(
            df,
            target_column=target_column,
            metadata_cols=registry_metadata_cols,
            exclude_cols=exclude_cols,
            split_pct_col=self.split_pct_col,
            original_names=original_names,
        )
        dataset = self._dataset_for(
            df,
            registry,
            unique_key=unique_key,
            split_group_key=split_group_key,
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

    def _normalize_key_columns(
        self, columns: tuple[str, ...], original_names: dict[str, str]
    ) -> tuple[str, ...]:
        raw_to_normalized = {raw: normalized for normalized, raw in original_names.items()}
        return tuple(
            raw_to_normalized.get(column, _normalize_column(column)) for column in columns
        )

    def _check_split_pct_collision(self, original_names: dict[str, str]) -> None:
        collisions = [
            raw
            for normalized, raw in original_names.items()
            if normalized == self.split_pct_col.lower()
        ]
        if collisions:
            raise DataError(
                f"source column(s) {collisions} collide with {self.split_pct_col}: the pipeline "
                "computes it from split_group_key — rename or remove the source column"
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
        unique_key: tuple[str, ...],
        split_group_key: tuple[str, ...],
        target_column: str,
        dataset_id: str = "unmaterialized",
    ) -> Dataset:
        source_identity = dict(self.spec.source.identity())
        source_identity["unique_key"] = list(unique_key)
        source_identity["split_group_key"] = list(split_group_key)
        component_hashes = ComponentHashes(
            source_identity=json_hash(source_identity),
            feature_registry=registry.content_hash(),
            data_content=dataframe_content_hash(df),
            schema=schema_hash(df),
        )
        identity_hash = json_hash(
            {
                "component_hashes": component_hashes.to_dict(),
                "unique_key": list(unique_key),
                "split_group_key": list(split_group_key),
                "n_columns": len(df.columns),
                "n_rows": len(df),
                "project_name": self.session.project_name,
                "split_pct_col": self.split_pct_col,
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
            split_pct_col=self.split_pct_col,
            unique_key=unique_key,
            split_group_key=split_group_key,
        )


def build_dataset(*, session: Session | None = None) -> LoadedDataset:
    active = _session(session)
    spec = active.config.require_data_spec()
    return spec.pipeline_cls(spec, active).run()


def materialize(
    *,
    refresh_data: bool = False,
    refresh_source: bool = False,
    include_rows: bool = True,
    session: Session | None = None,
) -> LoadedDataset | Dataset:
    """Attach to the active pinned dataset, or (re-)derive it on explicit refresh.

    refresh_source implies refresh_data: rebuilding layer 1 only matters if
    layer 2 is re-derived from it. Neither flag is ever passed by the agent
    loop — humans ask for refreshes (design §14).
    """
    refresh_data = refresh_data or refresh_source
    active = _session(session)
    with mlflow_client.bound_for(active, experiment_id=active.active_experiment_id):
        result = _materialize_bound(
            active=active,
            refresh_data=refresh_data,
            refresh_source=refresh_source,
            include_rows=include_rows,
        )
    return result


def _materialize_bound(
    *, active: Session, refresh_data: bool, refresh_source: bool, include_rows: bool
) -> LoadedDataset | Dataset:
    spec = active.config.require_data_spec()
    recipe = compute_recipe(spec, active)

    if not refresh_data:
        attached = _attach_active(active, recipe)
        if attached is not None:
            if not include_rows:
                return attached
            from automl.data.registry import load_dataset_by_id

            return load_dataset_by_id(attached.id, session=active)

    pipeline = spec.pipeline_cls(spec, active, refresh_source=refresh_source)
    loaded = pipeline.run()
    records = experiment_artifacts.list_dataset_records()
    existing = _record_for_identity(records, loaded.dataset.identity_hash)

    if existing is not None:
        # Content unchanged: attach; update the recorded recipe last-wins
        # (the user explicitly refreshed — "this recipe currently produces
        # this content" is honest provenance, design §3).
        dataset = replace(
            Dataset.from_dict(existing),
            recipe=recipe,
        )
        record_uri = experiment_artifacts.write_dataset_record(
            dataset.to_dict(), dataset_id=dataset.id
        )
        dataset = replace(dataset, record_uri=record_uri)
        mlflow_experiment.set_active_dataset(dataset.id, experiment_id=active.active_experiment_id)
        logger.info("content unchanged — attached to %s (recipe updated)", dataset.id)
        loaded = LoadedDataset(dataset=dataset, df=loaded.df, registry=loaded.registry)
        return loaded if include_rows else dataset

    dataset = replace(
        loaded.dataset,
        id=_next_dataset_id(records, loaded.dataset),
        recipe=recipe,
    )
    object_state = _dataset_object_state(dataset)
    if any(object_state.values()):
        present = [name for name, exists in object_state.items() if exists]
        raise DataError(
            f"GCS objects already present for new dataset {dataset.id}: {present} — "
            "refusing to overwrite; wipe state manually if this is intentional"
        )
    experiment_artifacts.write_dataset_frame(dataset.data_gcs_uri, loaded.df)
    experiment_artifacts.write_registry(dataset.registry_gcs_uri, loaded.registry.to_dataframe())
    record_uri = experiment_artifacts.write_dataset_record(
        dataset.to_dict(), dataset_id=dataset.id
    )
    dataset = replace(dataset, record_uri=record_uri)  # in memory only; the reader re-derives it
    _log_source_trace(dataset, spec.source.artifact_files(pipeline))
    mlflow_experiment.set_active_dataset(dataset.id, experiment_id=active.active_experiment_id)
    logger.info("minted %s and set it active", dataset.id)
    loaded = LoadedDataset(dataset=dataset, df=loaded.df, registry=loaded.registry)
    return loaded if include_rows else dataset


def _attach_active(active: Session, recipe: dict) -> Dataset | None:
    """The default fast path: resolve pointer -> record -> recipe compare."""
    active_id = mlflow_experiment.get_active_dataset(experiment_id=active.active_experiment_id)
    if active_id is None:
        return None
    record = experiment_artifacts.read_dataset_record(active_id)
    if record is None:
        return None
    dataset = Dataset.from_dict(record)
    drift = recipe_diff(dataset.recipe, recipe)
    if drift:
        logger.warning(
            "recipe drift: %s changed since %s — running against %s as pinned; "
            "pass --refresh-data to re-derive%s",
            ", ".join(drift),
            dataset.id,
            dataset.id,
            (
                " (base_table.sql changed: only --refresh-source rebuilds the base table)"
                if any(field.startswith("source.base_table_sql") for field in drift)
                else ""
            ),
        )
    else:
        logger.info("attached to %s (pinned)", dataset.id)
    return dataset


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


def _record_for_identity(records: list[dict], identity_hash: str) -> dict | None:
    for record in records:
        if record.get("identity_hash") == identity_hash:
            return record
    return None


def _next_dataset_id(records: list[dict], dataset: Dataset) -> str:
    max_version = 0
    for record in records:
        match = re.match(r"^v(\d+)_", str(record.get("id", "")))
        if match:
            max_version = max(max_version, int(match.group(1)))
    return f"v{max_version + 1}_{dataset.identity_hash.removeprefix('sha256:')[:8]}"


def _dataset_object_state(dataset: Dataset) -> dict[str, bool]:
    return {
        "data": gcs.blob_exists(dataset.data_gcs_uri),
        "registry": gcs.blob_exists(dataset.registry_gcs_uri),
    }


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
