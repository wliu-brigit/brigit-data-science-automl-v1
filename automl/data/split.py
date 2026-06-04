"""Deterministic split-bucket helpers, key normalization, and ingestion-edge checks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import pandas as pd

from automl.errors import DataError


Key: TypeAlias = str | Sequence[str]
SPLIT_PCT_COL = "SPLIT_PCT"


def _normalize_key(key: Key, *, field_name: str) -> tuple[str, ...]:
    """Normalize a key declaration to a sorted tuple of column names."""
    if isinstance(key, str):
        columns = (key,)
    else:
        try:
            columns = tuple(key)
        except TypeError as exc:
            raise ValueError(
                f"{field_name} must be a column name or a non-empty sequence of column names"
            ) from exc
    if not columns or any(not isinstance(column, str) or not column.strip() for column in columns):
        raise ValueError(f"{field_name} must contain non-empty column names")
    if len(set(columns)) != len(columns):
        raise ValueError(f"duplicate {field_name} columns are not allowed")
    return tuple(sorted(columns))


def add_split_pct(
    df: pd.DataFrame,
    *,
    split_group_key: tuple[str, ...],
    split_pct_col: str = SPLIT_PCT_COL,
    source_label: str = "data",
) -> pd.DataFrame:
    """Return ``df`` with deterministic 0..99 split buckets hashed from ``split_group_key``.

    The source must not already provide the column: a pre-existing split
    column has ambiguous provenance, so it is a loud error (symmetric with
    SnowflakeSource's injection collision error), never silently recomputed.
    """
    if split_pct_col in df.columns:
        raise DataError(
            f"{source_label} already provides a {split_pct_col} column; the pipeline computes "
            f"it from split_group_key — rename or remove the source column"
        )
    missing = [column for column in split_group_key if column not in df.columns]
    if missing:
        raise KeyError(
            f"split_group_key column(s) {missing} not in {source_label} columns: {list(df.columns)}"
        )
    split_pct = (
        pd.util.hash_pandas_object(df[list(split_group_key)], index=False)
        .mod(100)
        .astype("int64")
    )
    out = df.copy()
    out[split_pct_col] = split_pct.to_numpy()
    return out


def validate_unique_key(
    df: pd.DataFrame,
    *,
    unique_key: tuple[str, ...],
    source_label: str = "data",
) -> None:
    """Hard ingestion-edge check: unique_key columns present and duplicate-free."""
    missing = [column for column in unique_key if column not in df.columns]
    if missing:
        raise DataError(
            f"unique_key column(s) {missing} not in {source_label} columns: {list(df.columns)}"
        )
    duplicated = df.duplicated(subset=list(unique_key))
    if bool(duplicated.any()):
        examples = (
            df.loc[df.duplicated(subset=list(unique_key), keep=False), list(unique_key)]
            .head(5)
            .to_dict("records")
        )
        raise DataError(
            f"unique_key {unique_key} has {int(duplicated.sum())} duplicate row(s) in "
            f"{source_label}; examples: {examples}"
        )


def validate_split_pct(
    df: pd.DataFrame,
    *,
    split_pct_col: str = SPLIT_PCT_COL,
    source_label: str = "data",
) -> None:
    """Hard ingestion-edge check: split column present, integer, in 0–99."""
    if split_pct_col not in df.columns:
        raise DataError(
            f"{split_pct_col} missing from {source_label}; carry {split_pct_col} through "
            "from the base table"
        )
    series = df[split_pct_col]
    if not pd.api.types.is_integer_dtype(series):
        raise DataError(f"{split_pct_col} must be an integer column, got dtype {series.dtype}")
    if len(series) and not series.between(0, 99).all():
        raise DataError(f"{split_pct_col} values must be in 0–99")


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
    "Key",
    "SPLIT_PCT_COL",
    "add_split_pct",
    "split_report",
    "validate_split_pct",
    "validate_unique_key",
]
