"""GCS parquet data source."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from automl.data.sources.base import DataSource
from automl.data.split import HashKey, hash_key_columns
from automl.utils.io import gcs


@dataclass(frozen=True)
class GCSParquetSource(DataSource):
    gcs_uri: str
    hash_key: HashKey | None = None

    kind = "gcs_parquet"

    def __post_init__(self) -> None:
        gcs.parse_gcs_uri(self.gcs_uri)

    def load(
        self,
        *,
        project_dir: str | Path | None = None,
        nrows: int | None = None,
    ) -> pd.DataFrame:
        df = gcs.read_parquet(self.gcs_uri)
        if nrows is not None:
            return df.head(nrows)
        return df

    def identity(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "gcs_uri": self.gcs_uri,
            "hash_key": list(hash_key_columns(self.hash_key)),
        }


__all__ = ["GCSParquetSource"]
