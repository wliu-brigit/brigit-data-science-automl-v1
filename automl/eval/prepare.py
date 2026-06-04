"""Prepare durable eval dataset manifests."""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd

import automl.data as data
from automl.errors import EvalError
from automl.eval.eval_dataset import (
    Augmentation,
    EvalDataset,
    augmentation_root_uri,
    manifest_uri_for,
)
from automl.mlflow.experiment import eval_datasets
from automl.project import Session
from automl.project import session as active_project_session


def prepare_eval_dataset(
    *,
    session: Session | None = None,
    dataset_id: str | None = None,
    split: str | None = None,
    kind: str = "split_view",
    frame: pd.DataFrame | None = None,
    target_col: str | None = None,
    hash_key: Sequence[str] | None = None,
    provenance: Mapping[str, object] | None = None,
    overwrite: bool = False,
) -> tuple[EvalDataset, bool]:
    active = _session(session)
    if kind == "split_view":
        if not dataset_id or not split:
            raise ValueError("dataset_id and split are required for split_view eval datasets")
        return _prepare_split_view(active, dataset_id=dataset_id, split=split, overwrite=overwrite)
    if kind == "external":
        if frame is None:
            raise ValueError("frame is required for external eval datasets")
        return _prepare_external(
            active,
            frame=frame,
            target_col=target_col or active.config.target_column,
            hash_key=tuple(hash_key or ()),
            provenance=provenance,
            overwrite=overwrite,
        )
    raise ValueError(f"unsupported eval dataset kind {kind!r}")


def get_eval_dataset(eval_dataset_id: str, *, session: Session | None = None) -> EvalDataset:
    active = _session(session)
    return EvalDataset.from_dict(
        eval_datasets.read_manifest(manifest_uri_for(eval_dataset_id, session=active))
    )


def prepare_eval_augmentation(
    *,
    session: Session | None = None,
    eval_dataset_id: str,
    frame: pd.DataFrame,
    name: str,
    overwrite: bool = False,
) -> tuple[Augmentation, bool]:
    active = _session(session)
    base = get_eval_dataset(eval_dataset_id, session=active)
    from automl.eval._load import load_eval_dataset

    loaded = load_eval_dataset(eval_dataset_id, session=active)
    augmentation = Augmentation.create(
        session=active,
        eval_dataset_id=base.id,
        name=name,
        frame=frame,
        hash_key=base.hash_key,
    )
    data_exists = eval_datasets.blob_exists(augmentation.data_gcs_uri)
    manifest_exists = eval_datasets.blob_exists(augmentation.manifest_gcs_uri)
    if data_exists and manifest_exists and not overwrite:
        return Augmentation.from_dict(
            eval_datasets.read_manifest(augmentation.manifest_gcs_uri)
        ), True
    if data_exists != manifest_exists and not overwrite:
        raise EvalError(f"partial augmentation objects exist for {name!r}")

    _validate_augmentation_against_eval_frame(
        augmentation=augmentation,
        frame=frame,
        eval_frame=loaded.df,
        existing=_existing_augmentations(
            active, eval_dataset_id, skip=augmentation.manifest_gcs_uri
        ),
    )
    eval_datasets.write_frame(augmentation.data_gcs_uri, frame, overwrite=overwrite)
    eval_datasets.write_manifest(
        augmentation.manifest_gcs_uri,
        augmentation.to_dict(),
        overwrite=overwrite,
    )
    return augmentation, False


def _prepare_split_view(
    active: Session,
    *,
    dataset_id: str,
    split: str,
    overwrite: bool,
) -> tuple[EvalDataset, bool]:
    buckets = active.config.require_run_config().splits.resolve(split)
    parent = _dataset_by_id(data.list_datasets(session=active), dataset_id)
    recipe = EvalDataset.split_view(
        session=active,
        of_dataset_id=parent.id,
        split=split,
        split_pct_col=parent.split_pct_col,
        buckets=buckets,
        target_column=parent.target_column,
        hash_key=parent.hash_key,
    )
    if eval_datasets.blob_exists(recipe.manifest_gcs_uri) and not overwrite:
        return EvalDataset.from_dict(eval_datasets.read_manifest(recipe.manifest_gcs_uri)), True
    eval_datasets.write_manifest(recipe.manifest_gcs_uri, recipe.to_dict(), overwrite=overwrite)
    return recipe, False


def _prepare_external(
    active: Session,
    *,
    frame: pd.DataFrame,
    target_col: str,
    hash_key: Sequence[str],
    provenance: Mapping[str, object] | None,
    overwrite: bool,
) -> tuple[EvalDataset, bool]:
    recipe = EvalDataset.external(
        session=active,
        frame=frame,
        target_column=target_col,
        hash_key=hash_key,
        provenance=provenance,
    )
    if recipe.data_gcs_uri is None:
        raise EvalError("external eval dataset did not produce a data URI")
    manifest_exists = eval_datasets.blob_exists(recipe.manifest_gcs_uri)
    data_exists = eval_datasets.blob_exists(recipe.data_gcs_uri)
    if manifest_exists and data_exists and not overwrite:
        return EvalDataset.from_dict(eval_datasets.read_manifest(recipe.manifest_gcs_uri)), True
    if manifest_exists != data_exists and not overwrite:
        existing = [
            name
            for name, exists in {
                "manifest": manifest_exists,
                "data": data_exists,
            }.items()
            if exists
        ]
        raise EvalError(f"partial external eval dataset objects exist: {existing}")
    eval_datasets.write_frame(recipe.data_gcs_uri, frame, overwrite=overwrite)
    eval_datasets.write_manifest(recipe.manifest_gcs_uri, recipe.to_dict(), overwrite=overwrite)
    return recipe, False


def _dataset_by_id(index, dataset_id: str):
    for dataset in index.datasets:
        if dataset.id == dataset_id:
            return dataset
    raise KeyError(f"dataset {dataset_id!r} not found")


def _existing_augmentations(
    active: Session,
    eval_dataset_id: str,
    *,
    skip: str,
) -> tuple[Augmentation, ...]:
    augmentations = []
    for prefix in eval_datasets.list_prefixes(
        augmentation_root_uri(eval_dataset_id, session=active)
    ):
        manifest_uri = prefix.rstrip("/") + "/manifest.json"
        if manifest_uri == skip or not eval_datasets.blob_exists(manifest_uri):
            continue
        augmentations.append(Augmentation.from_dict(eval_datasets.read_manifest(manifest_uri)))
    return tuple(augmentations)


def _validate_augmentation_against_eval_frame(
    *,
    augmentation: Augmentation,
    frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    existing: Sequence[Augmentation],
) -> None:
    hash_key = list(augmentation.hash_key)
    base_rows = set(map(tuple, eval_frame.loc[:, hash_key].itertuples(index=False, name=None)))
    augmentation_rows = set(map(tuple, frame.loc[:, hash_key].itertuples(index=False, name=None)))
    missing_rows = sorted(augmentation_rows - base_rows)
    if missing_rows:
        raise ValueError(f"augmentation rows not present in eval dataset: {missing_rows[:5]}")
    value_columns = set(augmentation.columns) - set(augmentation.hash_key)
    base_overlap = sorted(value_columns & (set(eval_frame.columns) - set(augmentation.hash_key)))
    if base_overlap:
        raise ValueError(f"augmentation columns overlap eval dataset columns: {base_overlap}")
    existing_columns: set[str] = set()
    for item in existing:
        existing_columns.update(set(item.columns) - set(item.hash_key))
    existing_overlap = sorted(value_columns & existing_columns)
    if existing_overlap:
        raise ValueError(f"augmentation columns overlap existing augmentations: {existing_overlap}")


def _session(explicit: Session | None) -> Session:
    return explicit if explicit is not None else active_project_session()


__all__ = ["get_eval_dataset", "prepare_eval_augmentation", "prepare_eval_dataset"]
