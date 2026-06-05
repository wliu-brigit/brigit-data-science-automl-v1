"""Durable eval dataset identity and record value objects."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping, Sequence

import pandas as pd

from automl.mlflow import routing as mlflow_routing
from automl.project import Predicate, Session
from automl.utils.hashing import dataframe_content_hash, json_hash, schema_hash
from automl.utils.keys import normalize_key, validate_unique_key


_AUGMENTATION_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def compute_eval_dataset_identity(
    *,
    kind: str,
    target_column: str,
    unique_key: Sequence[str],
    of_dataset_id: str | None = None,
    predicate: Mapping[str, Any] | None = None,
    frame: pd.DataFrame | None = None,
) -> str:
    normalized_unique_key = _normalize_unique_key(unique_key)
    if kind == "split_view":
        if not of_dataset_id:
            raise ValueError("of_dataset_id required for split_view eval datasets")
        if not predicate:
            raise ValueError("predicate required for split_view eval datasets")
        # Identity hashes the serialized AST — what the split *means*.
        payload = {
            "schema_version": 1,
            "kind": kind,
            "of_dataset_id": of_dataset_id,
            "predicate": dict(predicate),
            "target_column": target_column,
            "unique_key": normalized_unique_key,
        }
    elif kind == "external":
        if frame is None:
            raise ValueError("frame required for external eval datasets")
        _validate_external_frame(frame, target_column=target_column, unique_key=normalized_unique_key)
        payload = {
            "schema_version": 1,
            "kind": kind,
            "target_column": target_column,
            "unique_key": normalized_unique_key,
            "schema_hash": schema_hash(frame),
            "content_hash": dataframe_content_hash(frame),
        }
    else:
        raise ValueError(f"unsupported eval dataset kind {kind!r}")
    digest = json_hash(payload).removeprefix("sha256:")[:12]
    return f"ev_{digest}"


def compute_augmentation_identity(
    eval_dataset_id: str,
    name: str,
    frame: pd.DataFrame,
    unique_key: Sequence[str],
) -> str:
    normalized_unique_key = _normalize_unique_key(unique_key)
    _validate_augmentation_name(name)
    _validate_augmentation_frame(frame, unique_key=normalized_unique_key)
    payload = {
        "schema_version": 1,
        "eval_dataset_id": eval_dataset_id,
        "name": name,
        "unique_key": normalized_unique_key,
        "schema_hash": schema_hash(frame),
        "content_hash": dataframe_content_hash(frame),
    }
    return json_hash(payload)


@dataclass(frozen=True)
class EvalDataset:
    id: str
    kind: str
    target_column: str
    unique_key: tuple[str, ...]
    gcs_bucket: str
    gcs_prefix: str
    project_name: str
    experiment_id: str
    dry_run: bool
    namespace: str
    of_dataset_id: str = ""
    split: str = ""
    predicate: Mapping[str, Any] | None = None
    content_hash: str = ""
    schema_hash: str = ""
    provenance: Mapping[str, Any] | None = None
    created_at: str = ""
    n_rows: int = 0
    n_columns: int = 0
    schema_version: int = 1

    @classmethod
    def split_view(
        cls,
        *,
        session: Session,
        of_dataset_id: str,
        split: str,
        predicate: Predicate,
        target_column: str,
        unique_key: Sequence[str],
    ) -> "EvalDataset":
        # The Predicate object converts to its AST exactly once, here at the
        # boundary; identity and the stored field carry the same mapping.
        predicate_ast = predicate.to_dict()
        normalized_unique_key = _normalize_unique_key(unique_key)
        dataset_id = compute_eval_dataset_identity(
            kind="split_view",
            of_dataset_id=of_dataset_id,
            predicate=predicate_ast,
            target_column=target_column,
            unique_key=normalized_unique_key,
        )
        route = _route_fields(session)
        return cls(
            id=dataset_id,
            kind="split_view",
            of_dataset_id=of_dataset_id,
            split=split,
            predicate=predicate_ast,
            target_column=target_column,
            unique_key=normalized_unique_key,
            created_at=_now(),
            **route,
        )

    @classmethod
    def external(
        cls,
        *,
        session: Session,
        frame: pd.DataFrame,
        target_column: str,
        unique_key: Sequence[str],
        provenance: Mapping[str, Any] | None = None,
    ) -> "EvalDataset":
        normalized_unique_key = _normalize_unique_key(unique_key)
        _validate_external_frame(frame, target_column=target_column, unique_key=normalized_unique_key)
        dataset_id = compute_eval_dataset_identity(
            kind="external",
            frame=frame,
            target_column=target_column,
            unique_key=normalized_unique_key,
        )
        route = _route_fields(session)
        return cls(
            id=dataset_id,
            kind="external",
            target_column=target_column,
            unique_key=normalized_unique_key,
            content_hash=dataframe_content_hash(frame),
            schema_hash=schema_hash(frame),
            provenance=dict(provenance or {}),
            created_at=_now(),
            n_rows=int(len(frame)),
            n_columns=int(len(frame.columns)),
            **route,
        )

    @property
    def route_prefix(self) -> str:
        return _route_prefix(
            gcs_prefix=self.gcs_prefix,
            namespace=self.namespace,
            dry_run=self.dry_run,
            project_name=self.project_name,
            experiment_id=self.experiment_id,
        )

    @property
    def record_gcs_uri(self) -> str:
        return f"gs://{self.gcs_bucket}/{self.route_prefix}/eval/datasets/{self.id}/eval_dataset.json"

    @property
    def data_gcs_uri(self) -> str | None:
        if self.kind != "external":
            return None
        return f"gs://{self.gcs_bucket}/{self.route_prefix}/eval/datasets/{self.id}/data.parquet"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvalDataset":
        # Forward-only: a split_view record must carry its predicate AST. A
        # record without one is pre-step-4 state (split_pct_col/buckets) —
        # fail here, loudly, instead of loading predicate=None and exploding
        # at use. Old state is disposable; re-prepare the eval dataset.
        if str(payload.get("kind", "")) == "split_view" and not payload.get("predicate"):
            raise ValueError(
                "split_view eval dataset record has no predicate AST — pre-step-4 "
                "(bucket-range) records are not loadable; re-prepare the eval dataset"
            )
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            id=str(payload["id"]),
            kind=str(payload["kind"]),
            target_column=str(payload["target_column"]),
            unique_key=tuple(str(item) for item in payload.get("unique_key", ())),
            gcs_bucket=str(payload.get("gcs_bucket", "")),
            gcs_prefix=str(payload.get("gcs_prefix", "")),
            project_name=str(payload.get("project_name", "")),
            experiment_id=str(payload.get("experiment_id", "")),
            dry_run=bool(payload.get("dry_run", False)),
            namespace=str(payload.get("namespace", "")),
            of_dataset_id=str(payload.get("of_dataset_id", "")),
            split=str(payload.get("split", "")),
            predicate=dict(payload["predicate"]) if payload.get("predicate") else None,
            content_hash=str(payload.get("content_hash", "")),
            schema_hash=str(payload.get("schema_hash", "")),
            provenance=dict(payload["provenance"]) if "provenance" in payload else None,
            created_at=str(payload.get("created_at", "")),
            n_rows=int(payload.get("n_rows", 0)),
            n_columns=int(payload.get("n_columns", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "id": self.id,
            "kind": self.kind,
            "target_column": self.target_column,
            "unique_key": list(self.unique_key),
            "gcs_bucket": self.gcs_bucket,
            "gcs_prefix": self.gcs_prefix,
            "project_name": self.project_name,
            "experiment_id": self.experiment_id,
            "dry_run": self.dry_run,
            "namespace": self.namespace,
            "created_at": self.created_at,
            "record_gcs_uri": self.record_gcs_uri,
            "data_gcs_uri": self.data_gcs_uri,
        }
        if self.kind == "split_view":
            payload.update(
                {
                    "of_dataset_id": self.of_dataset_id,
                    "split": self.split,
                    "predicate": dict(self.predicate or {}),
                }
            )
        elif self.kind == "external":
            payload.update(
                {
                    "schema_hash": self.schema_hash,
                    "content_hash": self.content_hash,
                    "provenance": dict(self.provenance or {}),
                    "n_rows": self.n_rows,
                    "n_columns": self.n_columns,
                }
            )
        return payload


@dataclass(frozen=True)
class Augmentation:
    eval_dataset_id: str
    name: str
    hash8: str
    content_hash: str
    schema_hash: str
    unique_key: tuple[str, ...]
    columns: tuple[str, ...]
    gcs_bucket: str
    gcs_prefix: str
    project_name: str
    experiment_id: str
    dry_run: bool
    namespace: str
    created_at: str = ""
    n_rows: int = 0
    n_columns: int = 0
    schema_version: int = 1

    @classmethod
    def create(
        cls,
        *,
        session: Session,
        eval_dataset_id: str,
        name: str,
        frame: pd.DataFrame,
        unique_key: Sequence[str],
    ) -> "Augmentation":
        normalized_unique_key = _normalize_unique_key(unique_key)
        _validate_augmentation_name(name)
        _validate_augmentation_frame(frame, unique_key=normalized_unique_key)
        identity = compute_augmentation_identity(
            eval_dataset_id,
            name,
            frame,
            normalized_unique_key,
        )
        route = _route_fields(session)
        return cls(
            eval_dataset_id=eval_dataset_id,
            name=name,
            hash8=identity.removeprefix("sha256:")[:8],
            content_hash=dataframe_content_hash(frame),
            schema_hash=schema_hash(frame),
            unique_key=normalized_unique_key,
            columns=tuple(str(column) for column in frame.columns),
            created_at=_now(),
            n_rows=int(len(frame)),
            n_columns=int(len(frame.columns)),
            **route,
        )

    @property
    def route_prefix(self) -> str:
        return _route_prefix(
            gcs_prefix=self.gcs_prefix,
            namespace=self.namespace,
            dry_run=self.dry_run,
            project_name=self.project_name,
            experiment_id=self.experiment_id,
        )

    @property
    def base_gcs_uri(self) -> str:
        return (
            f"gs://{self.gcs_bucket}/{self.route_prefix}/eval/datasets/"
            f"{self.eval_dataset_id}/augmentations/{self.name}__{self.hash8}"
        )

    @property
    def data_gcs_uri(self) -> str:
        return f"{self.base_gcs_uri}/data.parquet"

    @property
    def record_gcs_uri(self) -> str:
        return f"{self.base_gcs_uri}/augmentation.json"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Augmentation":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            eval_dataset_id=str(payload["eval_dataset_id"]),
            name=str(payload["name"]),
            hash8=str(payload["hash8"]),
            content_hash=str(payload["content_hash"]),
            schema_hash=str(payload["schema_hash"]),
            unique_key=tuple(str(item) for item in payload.get("unique_key", ())),
            columns=tuple(str(item) for item in payload.get("columns", ())),
            gcs_bucket=str(payload.get("gcs_bucket", "")),
            gcs_prefix=str(payload.get("gcs_prefix", "")),
            project_name=str(payload.get("project_name", "")),
            experiment_id=str(payload.get("experiment_id", "")),
            dry_run=bool(payload.get("dry_run", False)),
            namespace=str(payload.get("namespace", "")),
            created_at=str(payload.get("created_at", "")),
            n_rows=int(payload.get("n_rows", 0)),
            n_columns=int(payload.get("n_columns", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "eval_dataset_id": self.eval_dataset_id,
            "name": self.name,
            "hash8": self.hash8,
            "content_hash": self.content_hash,
            "schema_hash": self.schema_hash,
            "unique_key": list(self.unique_key),
            "columns": list(self.columns),
            "gcs_bucket": self.gcs_bucket,
            "gcs_prefix": self.gcs_prefix,
            "project_name": self.project_name,
            "experiment_id": self.experiment_id,
            "dry_run": self.dry_run,
            "namespace": self.namespace,
            "created_at": self.created_at,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "data_gcs_uri": self.data_gcs_uri,
            "record_gcs_uri": self.record_gcs_uri,
        }


def record_uri_for(eval_dataset_id: str, *, session: Session) -> str:
    route = _route_fields(session)
    prefix = _route_prefix(
        gcs_prefix=route["gcs_prefix"],
        namespace=route["namespace"],
        dry_run=route["dry_run"],
        project_name=route["project_name"],
        experiment_id=route["experiment_id"],
    )
    return f"gs://{route['gcs_bucket']}/{prefix}/eval/datasets/{eval_dataset_id}/eval_dataset.json"


def augmentation_root_uri(eval_dataset_id: str, *, session: Session) -> str:
    route = _route_fields(session)
    prefix = _route_prefix(
        gcs_prefix=route["gcs_prefix"],
        namespace=route["namespace"],
        dry_run=route["dry_run"],
        project_name=route["project_name"],
        experiment_id=route["experiment_id"],
    )
    return f"gs://{route['gcs_bucket']}/{prefix}/eval/datasets/{eval_dataset_id}/augmentations/"


def _validate_external_frame(
    frame: pd.DataFrame,
    *,
    target_column: str,
    unique_key: Sequence[str],
) -> None:
    if target_column not in frame.columns:
        raise ValueError(f"target column {target_column!r} missing from external eval frame")
    validate_unique_key(frame, unique_key=unique_key, source_label="external eval frame")


def _normalize_unique_key(unique_key: Sequence[str]) -> tuple[str, ...]:
    # The shared normalizer (utils.keys: sorted, blank/duplicate-free) so the
    # same composite key always normalizes — and hashes — identically on both
    # sides of the eval join (carried in from the step-1 review).
    return normalize_key(unique_key, field_name="unique_key")


def _validate_augmentation_name(name: str) -> None:
    if not isinstance(name, str) or not _AUGMENTATION_NAME_RE.fullmatch(name):
        raise ValueError("augmentation name must start lowercase and use lowercase letters, numbers, or underscores")


def _validate_augmentation_frame(frame: pd.DataFrame, *, unique_key: Sequence[str]) -> None:
    validate_unique_key(frame, unique_key=unique_key, source_label="augmentation frame")
    value_columns = [column for column in frame.columns if column not in set(unique_key)]
    if not value_columns:
        raise ValueError("augmentation frame must contain at least one non-unique-key column")


def _route_fields(session: Session) -> dict[str, Any]:
    return {
        "gcs_bucket": session.config.gcs_bucket,
        "gcs_prefix": session.config.gcs_prefix,
        "project_name": session.project_name,
        "experiment_id": session.active_experiment_id,
        "dry_run": session.dry_run,
        "namespace": session.namespace,
    }


def _route_prefix(
    *,
    gcs_prefix: str,
    namespace: str,
    dry_run: bool,
    project_name: str,
    experiment_id: str,
) -> str:
    return mlflow_routing.experiment_route_prefix_for(
        gcs_prefix=gcs_prefix,
        project_name=project_name,
        experiment_id=experiment_id,
        namespace=namespace,
        dry_run=dry_run,
    )


def _now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "Augmentation",
    "EvalDataset",
    "augmentation_root_uri",
    "compute_augmentation_identity",
    "compute_eval_dataset_identity",
    "record_uri_for",
]
