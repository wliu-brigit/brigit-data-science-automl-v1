"""Deterministic split-id helpers for data materialization."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import pandas as pd


HashKey: TypeAlias = str | Sequence[str]
ROW_FALLBACK_HASH_KEY = "__row_fallback__"


def hash_key_columns(hash_key: HashKey | None) -> tuple[str, ...]:
    """Normalize a hash-key declaration."""
    if hash_key is None:
        return (ROW_FALLBACK_HASH_KEY,)
    if isinstance(hash_key, str):
        columns = (hash_key,)
    else:
        try:
            columns = tuple(hash_key)
        except TypeError as exc:
            raise ValueError(
                "hash_key must be a column name, a non-empty sequence, or None"
            ) from exc
    if not columns or any(not isinstance(column, str) or not column.strip() for column in columns):
        raise ValueError("hash_key must contain non-empty column names")
    if len(set(columns)) != len(columns):
        raise ValueError("duplicate hash_key columns are not allowed")
    return tuple(sorted(columns))


def add_split_id(
    df: pd.DataFrame,
    *,
    hash_key: HashKey | None,
    split_id_col: str = "SPLITID",
    source_label: str = "data",
) -> pd.DataFrame:
    """Return ``df`` with deterministic 0..99 split buckets.

    When ``hash_key`` is omitted, the data layer uses a conservative fallback based on
    the full row content plus the loaded row index. This is deterministic for a
    stable source file, but users should provide stable business keys for durable
    cross-refresh split identity.
    """
    out = df.loc[:, [column for column in df.columns if column != split_id_col]].copy()
    if hash_key is None:
        split_ids = pd.util.hash_pandas_object(out, index=True).mod(100).astype("int64")
        out[split_id_col] = split_ids.to_numpy()
        return out

    columns = hash_key_columns(hash_key)
    missing = [column for column in columns if column not in out.columns]
    if missing:
        raise KeyError(
            f"hash_key column(s) {missing} not in {source_label} columns: {list(out.columns)}"
        )
    split_ids = (
        pd.util.hash_pandas_object(out[list(columns)], index=False)
        .mod(100)
        .astype("int64")
    )
    out[split_id_col] = split_ids.to_numpy()
    return out


def split_report(
    df: pd.DataFrame,
    *,
    split_id_col: str = "SPLITID",
) -> pd.DataFrame:
    if split_id_col not in df.columns:
        raise KeyError(f"split_report requires {split_id_col!r}")
    counts = df[split_id_col].value_counts().sort_index()
    return pd.DataFrame({"bucket": counts.index.astype(int), "rows": counts.to_numpy()})


__all__ = ["HashKey", "ROW_FALLBACK_HASH_KEY", "add_split_id", "hash_key_columns", "split_report"]
