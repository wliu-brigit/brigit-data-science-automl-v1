"""Feature registry primitives for the data path."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from automl.utils.hashing import json_hash


_NULL_SENTINELS = {
    "",
    " ",
    "nan",
    "NaN",
    "NAN",
    '"nan"',
    '"NaN"',
    '"Nan"',
    "none",
    "None",
    '"None"',
    "inf",
    "Inf",
    "INF",
    "Infinity",
    '"inf"',
    '"Inf"',
    '"Infinity"',
}

_BOOL_MAP = {
    "true": 1.0,
    "True": 1.0,
    "TRUE": 1.0,
    "false": 0.0,
    "False": 0.0,
    "FALSE": 0.0,
    True: 1.0,
    False: 0.0,
}


@dataclass
class FeatureEntry:
    name: str
    dtype: str
    original_name: str = ""
    null_pct: float = 0.0
    nunique: int = 0
    dominance_pct: float = 0.0
    available: bool = False
    feature: bool = False
    model: bool = False
    target: bool = False
    comments: str = ""
    derived: bool = False
    source_columns: tuple[str, ...] = ()


class FeatureRegistry:
    BOOL = "bool"
    NUM = "num"
    CAT = "cat"
    FLAGS = ("available", "feature", "model", "target", "derived")

    def __init__(self) -> None:
        self._entries: dict[str, FeatureEntry] = {}

    def build_from_df(
        self,
        df: pd.DataFrame,
        *,
        target_column: str,
        metadata_cols: Iterable[str] = (),
        exclude_cols: Iterable[str] = (),
        split_id_col: str = "SPLITID",
        original_names: dict[str, str] | None = None,
    ) -> "FeatureRegistry":
        metadata = set(metadata_cols)
        excluded = set(exclude_cols)
        for column in df.columns:
            is_target = column == target_column
            is_metadata = column in metadata or column == split_id_col
            is_excluded = column in excluded
            series = df[column]
            self._entries[column] = FeatureEntry(
                name=column,
                dtype=_dtype_label(series),
                original_name=(original_names or {}).get(column, column),
                null_pct=float(series.isna().mean()) if len(series) else 0.0,
                nunique=int(series.nunique(dropna=True)),
                dominance_pct=_dominance_pct(series),
                available=True,
                feature=not is_target and not is_metadata and not is_excluded,
                model=not is_target and not is_metadata and not is_excluded,
                target=is_target,
            )
        return self

    def get(self, name: str) -> FeatureEntry:
        return self._entries[name]

    @property
    def columns(self) -> list[str]:
        return sorted(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __repr__(self) -> str:
        flag_counts = ", ".join(
            f"{flag}={sum(1 for entry in self._entries.values() if getattr(entry, flag))}"
            for flag in self.FLAGS
        )
        return f"FeatureRegistry(total={len(self._entries)}, {flag_counts})"

    def add_derived(
        self,
        name: str,
        dtype: str,
        source_columns: tuple[str, ...] | list[str],
        *,
        model: bool = True,
        comments: str = "",
    ) -> None:
        if name in self._entries:
            raise ValueError(f"feature {name!r} already exists")
        missing = [column for column in source_columns if column not in self._entries]
        if missing:
            raise KeyError(f"source column(s) missing from registry: {missing}")
        self._entries[name] = FeatureEntry(
            name=name,
            dtype=dtype,
            available=True,
            feature=True,
            model=model,
            comments=comments,
            derived=True,
            source_columns=tuple(source_columns),
        )

    def get_by_flag(self, flag: str) -> list[str]:
        _check_flag(flag)
        return sorted(name for name, entry in self._entries.items() if getattr(entry, flag))

    def set_flag(self, columns: str | Iterable[str], flag: str, value: bool) -> None:
        _check_flag(flag)
        selected = [columns] if isinstance(columns, str) else list(columns)
        missing = [column for column in selected if column not in self._entries]
        if missing:
            raise KeyError(f"column(s) missing from registry: {missing}")
        for column in selected:
            setattr(self._entries[column], flag, bool(value))

    def get_by_dtype(self, dtype: str, flag: str | None = "available") -> list[str]:
        if dtype not in (self.BOOL, self.NUM, self.CAT):
            raise ValueError(f"Unknown dtype {dtype!r}. Valid values: 'bool', 'num', 'cat'")
        if flag is not None:
            _check_flag(flag)
        return sorted(
            name
            for name, entry in self._entries.items()
            if entry.dtype == dtype and (flag is None or getattr(entry, flag))
        )

    def select(self, df: pd.DataFrame, *, flag: str = "feature") -> pd.DataFrame:
        selected_cols = self.get_by_flag(flag)
        missing = [column for column in selected_cols if column not in df.columns]
        if missing:
            raise KeyError(
                f"FeatureRegistry.select: columns flagged {flag}=True missing "
                f"from input: {missing}"
            )
        return df[selected_cols]

    def cast(self, df: pd.DataFrame, *, inplace: bool = True) -> pd.DataFrame:
        out = df if inplace else df.copy()
        object_cols = out.select_dtypes(include="object").columns.tolist()
        if object_cols:
            out[object_cols] = out[object_cols].replace({sentinel: np.nan for sentinel in _NULL_SENTINELS})

        bool_cols = [
            column
            for column in out.columns
            if column in self._entries and self._entries[column].dtype == self.BOOL
        ]
        num_cols = [
            column
            for column in out.columns
            if column in self._entries and self._entries[column].dtype == self.NUM
        ]
        cat_cols = [
            column
            for column in out.columns
            if column in self._entries and self._entries[column].dtype == self.CAT
        ]

        if bool_cols:
            out[bool_cols] = (
                out[bool_cols]
                .apply(lambda series: series.map(lambda value: _BOOL_MAP.get(value, value)))
                .apply(pd.to_numeric, errors="coerce")
                .astype("float64")
            )
        if num_cols:
            out[num_cols] = out[num_cols].apply(pd.to_numeric, errors="coerce").astype("float64")
        if cat_cols:
            na_mask = out[cat_cols].isna()
            out[cat_cols] = out[cat_cols].astype(str).mask(na_mask)
        return out

    def add_comment(self, columns: str | Iterable[str], text: str) -> None:
        selected = [columns] if isinstance(columns, str) else list(columns)
        for column in selected:
            if column not in self._entries:
                continue
            existing = self._entries[column].comments
            self._entries[column].comments = f"{existing}\n{text}".lstrip("\n")

    def get_comment(self, name: str) -> str:
        entry = self._entries.get(name)
        return "" if entry is None else entry.comments

    def to_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for entry in self._entries.values():
            row = entry.__dict__.copy()
            row["source_columns"] = json.dumps(list(entry.source_columns))
            rows.append(row)
        columns = [
            "name",
            "dtype",
            "original_name",
            "null_pct",
            "nunique",
            "dominance_pct",
            "available",
            "feature",
            "model",
            "target",
            "comments",
            "derived",
            "source_columns",
        ]
        return pd.DataFrame(rows, columns=columns)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "FeatureRegistry":
        registry = cls()
        for _, row in df.iterrows():
            source_columns = row.get("source_columns", "[]")
            if isinstance(source_columns, str):
                parsed = json.loads(source_columns or "[]")
            else:
                parsed = []
            entry = FeatureEntry(
                name=str(row["name"]),
                dtype=str(row.get("dtype", "")),
                original_name=_str_or_empty(row.get("original_name", "")),
                null_pct=float(row.get("null_pct", 0.0) or 0.0),
                nunique=int(row.get("nunique", 0) or 0),
                dominance_pct=float(row.get("dominance_pct", 0.0) or 0.0),
                available=bool(row.get("available", False)),
                feature=bool(row.get("feature", False)),
                model=bool(row.get("model", False)),
                target=bool(row.get("target", False)),
                comments=_str_or_empty(row.get("comments", "")),
                derived=bool(row.get("derived", False)),
                source_columns=tuple(str(item) for item in parsed),
            )
            registry._entries[entry.name] = entry
        return registry

    def content_hash(self) -> str:
        frame = self.to_dataframe()
        for column in frame.select_dtypes(include="float").columns:
            frame[column] = frame[column].round(12)
        frame = frame.where(pd.notna(frame), "")
        ordered = frame.sort_values("name", kind="mergesort").reset_index(drop=True)
        return json_hash({"columns": list(ordered.columns), "records": ordered.to_dict("records")})


def _dtype_label(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return FeatureRegistry.BOOL
    if pd.api.types.is_numeric_dtype(series):
        return FeatureRegistry.NUM
    return FeatureRegistry.CAT


def _dominance_pct(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    counts = series.value_counts(dropna=False)
    if counts.empty:
        return 0.0
    return float(counts.iloc[0] / len(series))


def _str_or_empty(value: Any) -> str:
    return "" if pd.isna(value) else str(value)


def _check_flag(flag: str) -> None:
    if flag not in FeatureRegistry.FLAGS:
        raise ValueError(
            f"Unknown feature registry flag {flag!r}. "
            f"Valid flags: {', '.join(FeatureRegistry.FLAGS)}"
        )


__all__ = ["FeatureEntry", "FeatureRegistry"]
