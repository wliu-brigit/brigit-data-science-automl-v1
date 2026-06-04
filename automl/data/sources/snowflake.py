"""Snowflake source placeholder."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from automl.data.sources.base import DataSource
from automl.data.split import Key


@dataclass(frozen=True)
class SnowflakeSource(DataSource):
    base_table: str
    base_data_sql: str | Path
    training_data_sql: str | Path
    unique_key: Key
    split_group_key: Key | None = None

    kind = "snowflake"

    def load(
        self,
        *,
        project_dir: str | Path | None = None,
        nrows: int | None = None,
    ) -> pd.DataFrame:
        raise NotImplementedError(
            "SnowflakeSource is a pending source implementation; live Snowflake loading is not "
            "implemented in the new data layer yet."
        )

    def identity(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "base_table": self.base_table,
            "base_data_sql": str(self.base_data_sql),
            "training_data_sql": str(self.training_data_sql),
            "snowflake_database": os.environ.get("SNOWFLAKE_DATABASE", ""),
            "snowflake_schema": os.environ.get("SNOWFLAKE_SCHEMA", ""),
            "unique_key": list(self.unique_key_columns),
            "split_group_key": list(self.split_group_key_columns),
        }


__all__ = ["SnowflakeSource"]
