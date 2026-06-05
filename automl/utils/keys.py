"""Shared unique-key vocabulary: declaration normalization and frame checks.

The unique-key guarantee is cross-cutting — data hard-validates it at the
ingestion edge, eval re-validates it on both sides of its one-to-one joins —
so the one implementation lives down here in utils (moved 2026-06-04; five
hand-rolled copies had started to drift). Callers pass their domain's error
type via ``error_cls``; the logic itself never forks.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import pandas as pd

Key: TypeAlias = str | Sequence[str]


def normalize_key(key: Key, *, field_name: str = "unique_key") -> tuple[str, ...]:
    """Normalize a key declaration to a sorted tuple of column names.

    Sorted so the same composite key always normalizes — and hashes —
    identically wherever it is declared. Raises ValueError on bad
    declarations (empty, blank, duplicate, or non-string entries).
    """
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


def validate_unique_key(
    df: pd.DataFrame,
    *,
    unique_key: Sequence[str],
    source_label: str = "data",
    error_cls: type[Exception] = ValueError,
) -> None:
    """Hard edge check: unique_key columns present, non-null, duplicate-free.

    ``error_cls`` carries the calling domain's exception type (DataError at
    the data ingestion edge, ValueError in eval) over the one implementation.
    """
    missing = [column for column in unique_key if column not in df.columns]
    if missing:
        raise error_cls(
            f"unique_key column(s) {missing} not in {source_label} columns: {list(df.columns)}"
        )
    null_rows = df[list(unique_key)].isna().any(axis=1)
    if bool(null_rows.any()):
        raise error_cls(
            f"unique_key {tuple(unique_key)} has {int(null_rows.sum())} row(s) with null key "
            f"values in {source_label}; a stable row identifier must be non-null"
        )
    duplicated = df.duplicated(subset=list(unique_key))
    if bool(duplicated.any()):
        examples = (
            df.loc[df.duplicated(subset=list(unique_key), keep=False), list(unique_key)]
            .head(5)
            .to_dict("records")
        )
        raise error_cls(
            f"unique_key {tuple(unique_key)} has {int(duplicated.sum())} duplicate row(s) in "
            f"{source_label}; examples: {examples}"
        )


__all__ = ["Key", "normalize_key", "validate_unique_key"]
