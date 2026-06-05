"""Deterministic split-bucket helpers and ingestion-edge checks.

Key normalization and the unique-key frame check live in
``automl.utils.keys`` (shared with eval); this module binds them to the
data domain's error type and re-exports the vocabulary.
"""

from __future__ import annotations

import pandas as pd

from automl.errors import DataError
from automl.utils import keys as _keys
from automl.utils.keys import Key

SPLIT_PCT_COL = "SPLIT_PCT"


def add_split_pct(
    df: pd.DataFrame,
    *,
    split_group_key: tuple[str, ...],
    split_pct_col: str = SPLIT_PCT_COL,
    source_label: str = "data",
) -> pd.DataFrame:
    """Return ``df`` with deterministic 0..99 split buckets hashed from ``split_group_key``.

    Pure mechanism: collision detection (a source column that would shadow
    ``split_pct_col``) lives at the pipeline edge, where pre-standardization
    column names are known — one altitude for the check, not two.
    """
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
    """Hard ingestion-edge check: unique_key columns present, non-null, duplicate-free."""
    _keys.validate_unique_key(
        df, unique_key=unique_key, source_label=source_label, error_cls=DataError
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
    if bool(series.isna().any()):
        # nullable Int64 passes the dtype check, and between(...).all() skips NA —
        # without this check, NA-bucket rows silently vanish from every split slice.
        raise DataError(
            f"{split_pct_col} has {int(series.isna().sum())} missing values in {source_label}"
        )
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
