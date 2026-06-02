"""Local CSV data source."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from automl.data.sources.base import DataSource
from automl.data.split import HashKey, hash_key_columns


@dataclass(frozen=True)
class LocalCSVSource(DataSource):
    csv_path: str | Path
    hash_key: HashKey | None = None

    kind = "local_csv"

    def load(
        self,
        *,
        project_dir: str | Path | None = None,
        nrows: int | None = None,
    ) -> pd.DataFrame:
        path = Path(self.csv_path)
        csv_path = path if path.is_absolute() else Path(project_dir or Path.cwd()) / path
        return pd.read_csv(csv_path, nrows=nrows)

    def identity(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "csv_path": str(self.csv_path),
            "hash_key": list(hash_key_columns(self.hash_key)),
        }


__all__ = ["LocalCSVSource"]
