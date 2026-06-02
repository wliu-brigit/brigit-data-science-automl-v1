"""Experiment-scoped eval dataset storage helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd

from automl.errors import StorageError
from automl.utils.io import gcs


def read_manifest(uri: str) -> dict[str, Any]:
    try:
        return gcs.read_json(uri)
    except Exception as exc:
        raise StorageError(f"Failed to read eval manifest {uri!r}") from exc


def write_manifest(uri: str, payload: dict[str, Any], *, overwrite: bool = False) -> None:
    try:
        gcs.write_json(uri, payload, overwrite=overwrite)
    except Exception as exc:
        raise StorageError(f"Failed to write eval manifest {uri!r}") from exc


def read_frame(uri: str) -> pd.DataFrame:
    try:
        return gcs.read_parquet(uri)
    except Exception as exc:
        raise StorageError(f"Failed to read eval frame {uri!r}") from exc


def write_frame(uri: str, frame: pd.DataFrame, *, overwrite: bool = False) -> None:
    try:
        gcs.write_parquet(uri, frame, overwrite=overwrite)
    except Exception as exc:
        raise StorageError(f"Failed to write eval frame {uri!r}") from exc


def blob_exists(uri: str) -> bool:
    return gcs.blob_exists(uri)


def list_blob_names(uri: str) -> list[str]:
    return gcs.list_blob_names(uri)


def list_prefixes(uri: str) -> list[str]:
    return gcs.list_prefixes(uri)


__all__ = [
    "blob_exists",
    "list_blob_names",
    "list_prefixes",
    "read_frame",
    "read_manifest",
    "write_frame",
    "write_manifest",
]
