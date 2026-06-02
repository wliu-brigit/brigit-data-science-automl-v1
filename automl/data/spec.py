"""Project-level data specification."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from automl.data.sources.base import DataSource


def _coerce_str_tuple(value: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise TypeError(f"{field_name} must be a sequence of strings, not a string")
    out: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field_name}[{index}] must be a non-empty string")
        out.append(item)
    return tuple(out)


def _validate_threshold(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number in [0, 1]")
    if not (0.0 <= float(value) <= 1.0):
        raise ValueError(f"{field_name} must be in [0, 1]")


@dataclass(frozen=True)
class DataSpec:
    source: DataSource
    exclude_cols: Sequence[str] = ()
    metadata_cols: Sequence[str] = ()
    pipeline_cls: Any = None
    null_drop_threshold: float = 0.99
    constant_drop_threshold: float = 1.0
    dry_run_rows: int = 10_001

    def __post_init__(self) -> None:
        if not isinstance(self.source, DataSource):
            raise TypeError(f"source must be a DataSource, got {type(self.source).__name__}")
        object.__setattr__(self, "exclude_cols", _coerce_str_tuple(self.exclude_cols, "exclude_cols"))
        object.__setattr__(
            self, "metadata_cols", _coerce_str_tuple(self.metadata_cols, "metadata_cols")
        )
        _validate_threshold(self.null_drop_threshold, "null_drop_threshold")
        _validate_threshold(self.constant_drop_threshold, "constant_drop_threshold")
        if (
            isinstance(self.dry_run_rows, bool)
            or not isinstance(self.dry_run_rows, int)
            or self.dry_run_rows < 1
        ):
            raise ValueError("dry_run_rows must be a positive integer")
        if self.pipeline_cls is None:
            from automl.data.pipeline import DataPipeline

            object.__setattr__(self, "pipeline_cls", DataPipeline)
        else:
            from automl.data.pipeline import DataPipeline

            if not (isinstance(self.pipeline_cls, type) and issubclass(self.pipeline_cls, DataPipeline)):
                raise TypeError("pipeline_cls must be a DataPipeline subclass")


__all__ = ["DataSpec"]
