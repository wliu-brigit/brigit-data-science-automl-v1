"""Snowflake source placeholder."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from automl.data.sources.base import DataSource


@dataclass(frozen=True)
class SnowflakeSource(DataSource):
    base_table: str
    base_data_sql: str | Path
    training_data_sql: str | Path

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
            "hash_key": [],
        }


__all__ = ["SnowflakeSource"]
