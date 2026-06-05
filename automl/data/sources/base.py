"""Data source extension anchor."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from automl.utils.keys import Key, normalize_key

if TYPE_CHECKING:
    from automl.data.pipeline import DataPipeline


class DataSource(ABC):
    kind = "base"
    unique_key: Key
    split_group_key: Key | None = None

    @property
    def unique_key_columns(self) -> tuple[str, ...]:
        """The declared stable row identifier, normalized."""
        return normalize_key(self.unique_key, field_name="unique_key")

    @property
    def split_group_key_columns(self) -> tuple[str, ...]:
        """The key whose hash assigns split buckets; defaults to unique_key."""
        if self.split_group_key is None:
            return self.unique_key_columns
        return normalize_key(self.split_group_key, field_name="split_group_key")

    @abstractmethod
    def load(
        self,
        *,
        project_dir: str | Path | None = None,
        nrows: int | None = None,
        refresh_source: bool = False,
    ) -> pd.DataFrame:
        """Load raw rows; refresh_source asks the source to rebuild its upstream first."""

    @abstractmethod
    def identity(self) -> dict[str, Any]:
        """Return deterministic source identity fields."""

    def recipe_identity(self, *, project_dir: str | Path | None = None) -> dict[str, Any]:
        """Recipe-side identity: config-only, never touches the source.

        Defaults to identity(); sources whose identity references files by
        path override this to hash file *content* (SnowflakeSource, step 3).
        """
        del project_dir
        return self.identity()

    def artifact_files(self, pipeline: "DataPipeline") -> dict[str, Path]:
        """Return source trace artifacts to attach to project overview runs."""
        return {}


__all__ = ["DataSource"]
