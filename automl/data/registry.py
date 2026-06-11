"""Dataset registry read verbs."""

from __future__ import annotations

from dataclasses import replace

from automl.data.cache import cache_key, dataset_cache
from automl.data.contract import (
    validate_loaded_dataset,
    validate_trial_data_contract,
    verify_loaded_slice,
    verify_trial_tag_lineage,
)
from automl.data.dataset import Dataset, DatasetIndex, LoadedDataset, LoadedSlice
from automl.data.features import FeatureRegistry
from automl.data.selection import resolve_active_dataset
from automl.errors import DataError, StorageError
from automl.mlflow import client as mlflow_client
from automl.mlflow import experiment as mlflow_experiment
from automl.mlflow.trial import artifacts as trial_artifacts
from automl.mlflow.experiment import artifacts as experiment_artifacts
from automl.project import Predicate, Session
from automl.project import session as active_project_session
from automl.utils.io import gcs


def list_datasets(*, session: Session | None = None) -> DatasetIndex:
    active = _session(session)
    with mlflow_client.bound_for(active, experiment_id=active.active_experiment_id):
        records = experiment_artifacts.list_dataset_records()
        return DatasetIndex(
            datasets=tuple(Dataset.from_dict(record) for record in records),
            active_dataset_id=mlflow_experiment.get_active_dataset(
                experiment_id=active.active_experiment_id
            ),
        )


def load_dataset(
    *,
    split_name: str | None = None,
    predicate: Predicate | None = None,
    session: Session | None = None,
) -> LoadedDataset | LoadedSlice:
    dataset = resolve_active_dataset(session=session)
    return load_dataset_by_id(
        dataset.id,
        split_name=split_name,
        predicate=predicate,
        session=session,
    )


def load_dataset_by_id(
    dataset_id: str,
    *,
    split_name: str | None = None,
    predicate: Predicate | None = None,
    session: Session | None = None,
) -> LoadedDataset | LoadedSlice:
    if split_name is not None and predicate is not None:
        raise ValueError("split_name and predicate are mutually exclusive")
    active = _session(session)
    with mlflow_client.bound_for(active, experiment_id=active.active_experiment_id):
        record = experiment_artifacts.read_dataset_record(dataset_id)
    if record is None:
        raise KeyError(f"dataset {dataset_id!r} not found")
    dataset = Dataset.from_dict(record)
    registry_frame, df = _read_dataset_files(dataset)
    registry = FeatureRegistry.from_dataframe(registry_frame)
    loaded = LoadedDataset(dataset=dataset, df=df, registry=registry)
    try:
        validate_loaded_dataset(loaded, dataset)
    except DataError:
        # Defense in depth: the cached bytes failed the manifest check. Evict
        # so the next read re-populates from GCS instead of re-failing forever.
        evict_dataset_entry(dataset)
        raise

    resolved = _resolve_predicate(active, split_name=split_name, predicate=predicate)
    if resolved is None:
        return loaded
    sliced = df[resolved.mask(df)].reset_index(drop=True)
    return LoadedSlice(
        dataset=dataset,
        df=sliced,
        registry=registry,
        split_name=split_name,
        predicate=resolved,
    )


def load_dataset_by_trial(
    trial_id: str,
    *,
    split_name: str | None = None,
    predicate: Predicate | None = None,
    session: Session | None = None,
    strict: bool = True,
) -> LoadedDataset | LoadedSlice:
    if split_name is not None and predicate is not None:
        raise ValueError("split_name and predicate are mutually exclusive")
    active = _session(session)
    with mlflow_client.bound_for(active, experiment_id=active.active_experiment_id):
        run_id = mlflow_experiment.find_trial_run_id(
            trial_id,
            experiment_id=active.active_experiment_id,
        )
        contract = trial_artifacts.load_trial_data_contract(run_id)
        if strict:
            verify_trial_tag_lineage(contract, run_id)

        resolved_name = split_name
        if split_name is not None:
            if split_name not in contract.splits:
                available = sorted(contract.splits)
                raise KeyError(
                    f"split {split_name!r} not found; available contract splits: {available}"
                )
            resolved = Predicate.from_dict(contract.splits[split_name])
        else:
            resolved = predicate

        loaded = load_dataset_by_id(
            contract.dataset.id,
            predicate=resolved,
            session=active,
        )
        if strict:
            validate_trial_data_contract(contract, loaded.dataset)
        if isinstance(loaded, LoadedSlice) and resolved_name is not None:
            loaded = replace(loaded, split_name=resolved_name)
            slice_contract = contract.slice(resolved_name)
            if strict and slice_contract is not None:
                verify_loaded_slice(loaded, slice_contract)
        return loaded


def _session(explicit: Session | None) -> Session:
    return explicit if explicit is not None else active_project_session()


def _resolve_predicate(
    active: Session,
    *,
    split_name: str | None,
    predicate: Predicate | None,
) -> Predicate | None:
    if split_name is None and predicate is None:
        return None
    if split_name is not None:
        return active.config.require_run_config().splits.resolve(split_name)
    return predicate


def _read_dataset_files(dataset: Dataset):
    """Read registry + frame through the local content-addressed cache.

    Non-GCS URIs (and records without a content hash) bypass the cache and
    use the direct artifact readers.
    """
    if not (
        gcs.is_gcs_uri(dataset.data_gcs_uri)
        and dataset.component_hashes.data_content
    ):
        return (
            experiment_artifacts.read_registry(dataset.registry_gcs_uri),
            experiment_artifacts.read_dataset_frame(dataset.data_gcs_uri),
        )
    import pandas as pd

    cache = dataset_cache()
    key = cache_key(dataset)
    try:
        data_path = cache.get_or_populate(
            key,
            "data.parquet",
            lambda tmp: gcs.download_to_file(dataset.data_gcs_uri, tmp),
        )
        registry_path = cache.get_or_populate(
            key,
            "feature_registry.csv",
            lambda tmp: gcs.download_to_file(dataset.registry_gcs_uri, tmp),
        )
        return pd.read_csv(registry_path), pd.read_parquet(data_path)
    except Exception as exc:
        raise StorageError(f"Failed to read dataset {dataset.id!r}") from exc


def evict_dataset_entry(dataset: Dataset) -> bool:
    """Drop a dataset's cached bytes (used on manifest-verification failure)."""
    return dataset_cache().remove(cache_key(dataset))


__all__ = ["list_datasets", "load_dataset", "load_dataset_by_id", "load_dataset_by_trial"]
