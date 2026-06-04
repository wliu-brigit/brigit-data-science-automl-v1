"""GCS parquet data source."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from automl.data.sources.base import DataSource
from automl.data.split import Key
from automl.utils.io import gcs


@dataclass(frozen=True)
class GCSParquetSource(DataSource):
    gcs_uri: str
    unique_key: Key
    split_group_key: Key | None = None

    kind = "gcs_parquet"

    def __post_init__(self) -> None:
        gcs.parse_gcs_uri(self.gcs_uri)
        self.unique_key_columns  # validate declarations at construction
        self.split_group_key_columns

    def load(
        self,
        *,
        project_dir: str | Path | None = None,
        nrows: int | None = None,
        refresh_source: bool = False,
    ) -> pd.DataFrame:
        del refresh_source  # layer-1 verbs are no-ops for file sources
        df = gcs.read_parquet(self.gcs_uri)
        if nrows is not None:
            return df.head(nrows)
        return df

    def identity(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "gcs_uri": self.gcs_uri,
            "unique_key": list(self.unique_key_columns),
            "split_group_key": list(self.split_group_key_columns),
        }


__all__ = ["GCSParquetSource"]
