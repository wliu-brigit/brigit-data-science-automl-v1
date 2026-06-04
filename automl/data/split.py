"""Deterministic split-bucket helpers for data materialization."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import pandas as pd


HashKey: TypeAlias = str | Sequence[str]
ROW_FALLBACK_HASH_KEY = "__row_fallback__"
SPLIT_PCT_COL = "SPLIT_PCT"


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


def add_split_pct(
    df: pd.DataFrame,
    *,
    hash_key: HashKey | None,
    split_pct_col: str = SPLIT_PCT_COL,
    source_label: str = "data",
) -> pd.DataFrame:
    """Return ``df`` with deterministic 0..99 split buckets in ``split_pct_col``."""
    out = df.loc[:, [column for column in df.columns if column != split_pct_col]].copy()
    if hash_key is None:
        split_pct = pd.util.hash_pandas_object(out, index=True).mod(100).astype("int64")
        out[split_pct_col] = split_pct.to_numpy()
        return out

    columns = hash_key_columns(hash_key)
    missing = [column for column in columns if column not in out.columns]
    if missing:
        raise KeyError(
            f"hash_key column(s) {missing} not in {source_label} columns: {list(out.columns)}"
        )
    split_pct = (
        pd.util.hash_pandas_object(out[list(columns)], index=False)
        .mod(100)
        .astype("int64")
    )
    out[split_pct_col] = split_pct.to_numpy()
    return out


def split_report(
    df: pd.DataFrame,
    *,
    split_pct_col: str = SPLIT_PCT_COL,
) -> pd.DataFrame:
    if split_pct_col not in df.columns:
        raise KeyError(f"split_report requires {split_pct_col!r}")
    counts = df[split_pct_col].value_counts().sort_index()
    return pd.DataFrame({"bucket": counts.index.astype(int), "rows": counts.to_numpy()})


__all__ = [
    "HashKey",
    "ROW_FALLBACK_HASH_KEY",
    "SPLIT_PCT_COL",
    "add_split_pct",
    "hash_key_columns",
    "split_report",
]
