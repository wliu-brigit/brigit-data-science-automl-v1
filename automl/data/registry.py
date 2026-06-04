"""Dataset registry read verbs."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from automl.data.contract import (
    validate_loaded_dataset,
    validate_trial_data_contract,
    verify_loaded_slice,
    verify_trial_tag_lineage,
)
from automl.data.dataset import Dataset, DatasetIndex, LoadedDataset, LoadedSlice
from automl.data.features import FeatureRegistry
from automl.errors import DataError
from automl.mlflow import client as mlflow_client
from automl.mlflow import experiment as mlflow_experiment
from automl.mlflow.trial import artifacts as trial_artifacts
from automl.mlflow.experiment import artifacts as experiment_artifacts
from automl.project import Session, Splits
from automl.project import session as active_project_session


def list_datasets(*, session: Session | None = None) -> DatasetIndex:
    active = _session(session)
    with mlflow_client.bound_for(active, experiment_id=active.active_experiment_id):
        index = DatasetIndex.from_dict(experiment_artifacts.read_dataset_index())
        return DatasetIndex(
            datasets=index.datasets,
            active_dataset_id=mlflow_experiment.get_active_dataset(
                experiment_id=active.active_experiment_id
            ),
            schema_version=index.schema_version,
        )


def load_dataset(
    *,
    split_name: str | None = None,
    split_range: tuple[int, int] | tuple[tuple[int, int], ...] | None = None,
    session: Session | None = None,
) -> LoadedDataset | LoadedSlice:
    index = list_datasets(session=session)
    dataset = index.active or (index.datasets[-1] if index.datasets else None)
    if dataset is None:
        raise DataError("no active dataset is available")
    return load_dataset_by_id(
        dataset.id,
        split_name=split_name,
        split_range=split_range,
        session=session,
    )


def load_dataset_by_id(
    dataset_id: str,
    *,
    split_name: str | None = None,
    split_range: tuple[int, int] | tuple[tuple[int, int], ...] | None = None,
    session: Session | None = None,
) -> LoadedDataset | LoadedSlice:
    if split_name is not None and split_range is not None:
        raise ValueError("split_name and split_range are mutually exclusive")
    active = _session(session)
    index = list_datasets(session=active)
    indexed = _dataset_by_id(index, dataset_id)
    manifest = experiment_artifacts.read_dataset_manifest(indexed.manifest_gcs_uri)
    dataset = Dataset.from_dict(manifest)
    registry_frame = experiment_artifacts.read_registry(dataset.registry_gcs_uri)
    registry = FeatureRegistry.from_dataframe(registry_frame)
    df = experiment_artifacts.read_dataset_frame(dataset.data_gcs_uri)
    loaded = LoadedDataset(dataset=dataset, df=df, registry=registry)
    validate_loaded_dataset(loaded, dataset)

    ranges = _resolve_ranges(active, split_name=split_name, split_range=split_range)
    if ranges is None:
        return loaded
    buckets = set(_buckets(ranges))
    sliced = df[df[dataset.split_pct_col].isin(buckets)].reset_index(drop=True)
    return LoadedSlice(
        dataset=dataset,
        df=sliced,
        registry=registry,
        split_name=split_name,
        split_ranges=ranges,
    )


def load_dataset_by_trial(
    trial_id: str,
    *,
    split_name: str | None = None,
    split_range: tuple[int, int] | tuple[tuple[int, int], ...] | None = None,
    session: Session | None = None,
    strict: bool = True,
) -> LoadedDataset | LoadedSlice:
    if split_name is not None and split_range is not None:
        raise ValueError("split_name and split_range are mutually exclusive")
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
            ranges = contract.splits[split_name]
        else:
            ranges = _normalize_split_range(split_range) if split_range is not None else None

        loaded = load_dataset_by_id(
            contract.dataset.id,
            split_range=ranges,
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


def _dataset_by_id(index: DatasetIndex, dataset_id: str) -> Dataset:
    for dataset in index.datasets:
        if dataset.id == dataset_id:
            return dataset
    raise KeyError(f"dataset {dataset_id!r} not found")


def _resolve_ranges(
    active: Session,
    *,
    split_name: str | None,
    split_range: tuple[int, int] | tuple[tuple[int, int], ...] | None,
) -> tuple[tuple[int, int], ...] | None:
    if split_name is None and split_range is None:
        return None
    if split_name is not None:
        return active.config.require_run_config().splits.resolve(split_name)
    return _normalize_split_range(split_range)


def _normalize_split_range(
    value: tuple[int, int] | tuple[tuple[int, int], ...] | None,
) -> tuple[tuple[int, int], ...]:
    if value is None:
        raise ValueError("split_range required")
    if len(value) == 2 and all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        value = (value,)  # type: ignore[assignment]
    return Splits({"slice": value}).resolve("slice")  # type: ignore[arg-type]


def _buckets(ranges: Iterable[tuple[int, int]]) -> frozenset[int]:
    buckets: set[int] = set()
    for low, high in ranges:
        buckets.update(range(low, high))
    return frozenset(buckets)


__all__ = ["list_datasets", "load_dataset", "load_dataset_by_id", "load_dataset_by_trial"]
