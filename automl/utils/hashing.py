"""Deterministic hashing primitives for JSON values and pandas DataFrames."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def json_hash(value: Any) -> str:
    """Return a stable ``sha256:...`` hash for a JSON-serializable value."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return _sha256_text(payload)


def dataframe_content_hash(df: pd.DataFrame) -> str:
    """Hash ordered columns, dtype strings, and row content for a DataFrame."""
    payload = {
        "columns": list(df.columns),
        "dtypes": [str(dtype) for dtype in df.dtypes],
        "row_hashes": pd.util.hash_pandas_object(df, index=False).astype("uint64").tolist(),
    }
    return json_hash(payload)


def schema_hash(df: pd.DataFrame) -> str:
    """Hash ordered column names and dtype strings for a DataFrame schema."""
    payload = {
        "columns": list(df.columns),
        "dtypes": [str(dtype) for dtype in df.dtypes],
    }
    return json_hash(payload)


__all__ = ["dataframe_content_hash", "json_hash", "schema_hash"]
