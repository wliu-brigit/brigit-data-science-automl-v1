"""Eval dataset loading delegated to durable records and the data domain."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import automl.data as data
from automl.eval.eval_dataset import (
    Augmentation,
    EvalDataset,
    augmentation_root_uri,
    record_uri_for,
)
from automl.mlflow.experiment import eval_datasets
from automl.project import Session
from automl.project import session as active_project_session
from automl.utils.hashing import dataframe_content_hash, schema_hash


@dataclass(frozen=True)
class LoadedEvalDataset:
    df: pd.DataFrame
    dataset: EvalDataset
    target_column: str
    unique_key: tuple[str, ...]
    row_ids: pd.DataFrame
    registry: object | None = None


def load_eval_dataset(eval_dataset_id: str, *, session: Session | None = None) -> LoadedEvalDataset:
    active = _session(session)
    record = eval_datasets.read_record(record_uri_for(eval_dataset_id, session=active))
    recipe = EvalDataset.from_dict(record)
    if recipe.kind == "split_view":
        loaded = data.load_dataset_by_id(
            recipe.of_dataset_id,
            split_range=recipe.buckets,
            session=active,
        )
        frame = loaded.df.reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"eval dataset {eval_dataset_id!r} is empty")
        return LoadedEvalDataset(
            df=frame,
            dataset=recipe,
            target_column=recipe.target_column,
            unique_key=recipe.unique_key,
            row_ids=frame.loc[:, list(recipe.unique_key)].reset_index(drop=True),
            registry=getattr(loaded, "registry", None),
        )
    if recipe.kind == "external":
        if recipe.data_gcs_uri is None:
            raise ValueError(f"external eval dataset {eval_dataset_id!r} is missing data URI")
        frame = eval_datasets.read_frame(recipe.data_gcs_uri).reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"eval dataset {eval_dataset_id!r} is empty")
        _validate_external_payload(recipe, frame)
        return LoadedEvalDataset(
            df=frame,
            dataset=recipe,
            target_column=recipe.target_column,
            unique_key=recipe.unique_key,
            row_ids=frame.loc[:, list(recipe.unique_key)].reset_index(drop=True),
        )
    raise ValueError(f"unsupported eval dataset kind {recipe.kind!r}")


def _validate_external_payload(recipe: EvalDataset, frame: pd.DataFrame) -> None:
    required_columns = (*recipe.unique_key, recipe.target_column)
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"external eval dataset is missing record column(s): {missing}")
    if frame.duplicated(subset=list(recipe.unique_key), keep=False).any():
        raise ValueError("external eval dataset contains duplicate unique_key rows")
    actual_schema_hash = schema_hash(frame)
    if recipe.schema_hash and actual_schema_hash != recipe.schema_hash:
        raise ValueError(
            "external eval dataset schema_hash mismatch: "
            f"expected {recipe.schema_hash}, got {actual_schema_hash}"
        )
    actual_content_hash = dataframe_content_hash(frame)
    if recipe.content_hash and actual_content_hash != recipe.content_hash:
        raise ValueError(
            "external eval dataset content_hash mismatch: "
            f"expected {recipe.content_hash}, got {actual_content_hash}"
        )


def load_eval_augmentations(
    eval_dataset_id: str,
    *,
    names: tuple[str, ...],
    session: Session | None = None,
) -> tuple[dict[str, pd.DataFrame], tuple[dict[str, str], ...]]:
    active = _session(session)
    found: dict[str, list[Augmentation]] = {name: [] for name in names}
    for prefix in eval_datasets.list_prefixes(
        augmentation_root_uri(eval_dataset_id, session=active)
    ):
        record_uri = prefix.rstrip("/") + "/augmentation.json"
        if not eval_datasets.blob_exists(record_uri):
            continue
        augmentation = Augmentation.from_dict(eval_datasets.read_record(record_uri))
        if augmentation.name in found:
            found[augmentation.name].append(augmentation)
    missing = [name for name, items in found.items() if not items]
    if missing:
        raise ValueError(f"augmentations not published on eval dataset: {missing}")

    frames: dict[str, pd.DataFrame] = {}
    used = []
    for name in names:
        latest = sorted(found[name], key=lambda item: item.created_at)[-1]
        frame = eval_datasets.read_frame(latest.data_gcs_uri)
        _validate_augmentation_payload(latest, frame)
        frames[name] = frame
        used.append(
            {
                "name": latest.name,
                "hash8": latest.hash8,
                "data_uri": latest.data_gcs_uri,
                "record_uri": latest.record_gcs_uri,
            }
        )
    return frames, tuple(used)


def _validate_augmentation_payload(augmentation: Augmentation, frame: pd.DataFrame) -> None:
    missing = [column for column in augmentation.columns if column not in frame.columns]
    if missing:
        raise ValueError(f"augmentation is missing record column(s): {missing}")
    if frame.duplicated(subset=list(augmentation.unique_key), keep=False).any():
        raise ValueError("augmentation contains duplicate unique_key rows")
    actual_schema_hash = schema_hash(frame)
    if actual_schema_hash != augmentation.schema_hash:
        raise ValueError(
            "augmentation schema_hash mismatch: "
            f"expected {augmentation.schema_hash}, got {actual_schema_hash}"
        )
    actual_content_hash = dataframe_content_hash(frame)
    if actual_content_hash != augmentation.content_hash:
        raise ValueError(
            "augmentation content_hash mismatch: "
            f"expected {augmentation.content_hash}, got {actual_content_hash}"
        )


def _session(explicit: Session | None) -> Session:
    return explicit if explicit is not None else active_project_session()


__all__ = ["LoadedEvalDataset", "load_eval_augmentations", "load_eval_dataset"]
