"""Data source extension anchor."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from automl.data.split import HashKey

if TYPE_CHECKING:
    from automl.data.pipeline import DataPipeline


class DataSource(ABC):
    kind = "base"
    hash_key: HashKey | None = None

    @abstractmethod
    def load(self, *, project_dir: str | Path | None = None, nrows: int | None = None) -> pd.DataFrame:
        """Load raw rows into a DataFrame."""

    @abstractmethod
    def identity(self) -> dict[str, Any]:
        """Return deterministic source identity fields."""

    def artifact_files(self, pipeline: "DataPipeline") -> dict[str, Path]:
        """Return source trace artifacts to attach to project overview runs."""
        return {}


__all__ = ["DataSource"]
