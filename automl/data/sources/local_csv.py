"""Local CSV data source."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from automl.data.sources.base import DataSource
from automl.data.split import Key


@dataclass(frozen=True)
class LocalCSVSource(DataSource):
    csv_path: str | Path
    unique_key: Key
    split_group_key: Key | None = None

    kind = "local_csv"

    def load(
        self,
        *,
        project_dir: str | Path | None = None,
        nrows: int | None = None,
        refresh_source: bool = False,
    ) -> pd.DataFrame:
        del refresh_source  # layer-1 verbs are no-ops for file sources
        path = Path(self.csv_path)
        csv_path = path if path.is_absolute() else Path(project_dir or Path.cwd()) / path
        return pd.read_csv(csv_path, nrows=nrows)

    def identity(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "csv_path": str(self.csv_path),
            "unique_key": list(self.unique_key_columns),
            "split_group_key": list(self.split_group_key_columns),
        }


__all__ = ["LocalCSVSource"]
