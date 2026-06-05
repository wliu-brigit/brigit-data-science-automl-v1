"""Snowflake data source: harness-owned DDL over a project-owned SELECT."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from automl.data.sources.base import DataSource
from automl.data.split import Key, SPLIT_PCT_COL
from automl.errors import DataError
from automl.utils.io import snowflake as sf

if TYPE_CHECKING:
    from automl.data.pipeline import DataPipeline


@dataclass(frozen=True)
class SnowflakeSource(DataSource):
    base_table: str
    base_table_sql: str | Path        # the SELECT defining the base data
    training_data_sql: str | Path     # the SELECT pulling training rows
    unique_key: Key
    split_group_key: Key | None = None

    kind = "snowflake"
    provides_split_pct = True         # SPLIT_PCT arrives frozen from the base table

    # --- substitutions -------------------------------------------------
    def _database(self) -> str:
        return os.environ.get("SNOWFLAKE_DATABASE", "")

    def _schema(self) -> str:
        return os.environ.get("SNOWFLAKE_SCHEMA", "")

    def _qualified_table(self) -> str:
        return f"{self._database()}.{self._schema()}.{self.base_table}"

    def _render(self, sql_path: str | Path, project_dir: str | Path | None) -> str:
        path = Path(sql_path)
        resolved = path if path.is_absolute() else Path(project_dir or Path.cwd()) / path
        text = resolved.read_text(encoding="utf-8")
        # Explicit replace, NOT str.format: Snowflake SQL legitimately
        # contains literal braces (OBJECT_CONSTRUCT('{...}'), semi-structured
        # paths) which str.format turns into an opaque KeyError. Only the
        # three documented substitutions are special. No case-mangling of
        # identifiers (the old implementation lowercased env values; dropped
        # as surprising — design §9).
        return (
            text.replace("{database}", self._database())
            .replace("{schema}", self._schema())
            .replace("{base_table}", self.base_table)
        )

    # --- DDL generation (design §9) -------------------------------------
    def generated_ddl(self, *, project_dir: str | Path | None = None) -> str:
        body = self._render(self.base_table_sql, project_dir).strip().rstrip(";").strip()
        statement = _scrub_sql(body)
        first_token = statement.split(None, 1)[0].upper() if statement.split() else ""
        if first_token not in ("SELECT", "WITH"):
            raise DataError(
                "base_table.sql must be a single SELECT (or WITH) statement; the harness "
                f"owns the CREATE — got a statement starting {first_token!r}"
            )
        if ";" in statement:
            raise DataError("base_table.sql must be a single SELECT statement (found ';')")
        if re.search(rf"\b{SPLIT_PCT_COL}\b", statement, re.IGNORECASE):
            raise DataError(
                f"base_table.sql already emits {SPLIT_PCT_COL}; the harness injects it from "
                "split_group_key — remove the column from the SELECT (one declaration only)"
            )
        key_args = ", ".join(f"t.{column}" for column in self.split_group_key_columns)
        return (
            f"CREATE OR REPLACE TABLE {self._qualified_table()} AS\n"
            f"SELECT t.*, MOD(ABS(HASH({key_args})), 100) AS {SPLIT_PCT_COL}\n"
            f"FROM (\n{body}\n) t"
        )

    # --- load (design §4 steps 2a-2d) -------------------------------------
    def load(
        self,
        *,
        project_dir: str | Path | None = None,
        nrows: int | None = None,
        refresh_source: bool = False,
    ) -> pd.DataFrame:
        # 2a/2b: ensure layer 1 exists — bootstrap when missing (nothing to
        # destroy), rebuild only on the explicit flag.
        if not self._table_exists() or refresh_source:
            sf.execute(self.generated_ddl(project_dir=project_dir))
        # 2c: empirical content check against the actual table — any table
        # satisfying the invariant is valid, whoever built it (design §4).
        self._check_split_invariant()
        # 2d: pull training rows; dry-run is a deterministic bucket sample.
        sql = self._render(self.training_data_sql, project_dir).strip().rstrip(";")
        if nrows is not None:
            sql = self._dry_run_sql(sql, nrows)
        return sf.fetch_df(sql)

    def _table_exists(self) -> bool:
        # Identifiers interpolated below are config/env-owned (base_table,
        # database, schema from the recipe), not user input — no injection
        # surface beyond what the project author already controls.
        # UPPER on both sides: Snowflake folds unquoted identifiers to
        # uppercase in INFORMATION_SCHEMA, so a lowercase env value must not
        # make this miss (a miss silently rebuilds the base table every load).
        row = sf.fetch_one(
            f"SELECT 1 FROM {self._database()}.INFORMATION_SCHEMA.TABLES "
            f"WHERE UPPER(TABLE_SCHEMA) = UPPER('{self._schema()}') "
            f"AND UPPER(TABLE_NAME) = UPPER('{self.base_table}')"
        )
        return row is not None

    def _check_split_invariant(self) -> None:
        key_args = ", ".join(self.split_group_key_columns)
        row = sf.fetch_one(
            f"SELECT 1 FROM {self._qualified_table()} "
            f"WHERE {SPLIT_PCT_COL} IS DISTINCT FROM MOD(ABS(HASH({key_args})), 100) LIMIT 1"
        )
        if row is not None:
            raise DataError(
                f"{self._qualified_table()} has {SPLIT_PCT_COL} values that do not match "
                f"split_group_key {self.split_group_key_columns} — the stored buckets are "
                "stale (the key changed, or the table was built out-of-band). Rebuild "
                "explicitly with --refresh-source; the harness never auto-rebuilds."
            )

    def _dry_run_sql(self, training_sql: str, nrows: int) -> str:
        # Count the training query, not the base table: training SQL may
        # filter or downsample, and the bucket fraction must size the dry-run
        # sample against the rows actually pulled (a base-table count made a
        # 39k pull look like 10.7M rows and clamped the sample to one bucket).
        row = sf.fetch_one(f"SELECT COUNT(*) FROM (\n{training_sql}\n)")
        total = int(row[0]) if row and row[0] else 0
        if total <= 0:
            buckets = 100
        else:
            buckets = max(1, min(100, round(100 * nrows / total)))
        # Whole hash-buckets trade exactness for determinism: the same sample
        # every run, so dry-run identity is stable (design §4). The bucket
        # hash is over unique_key, NOT SPLIT_PCT: splits cut on SPLIT_PCT, so
        # a prefix of it lands entirely inside one split (empty test
        # partition); an independent hash samples uniformly across all splits.
        key_args = ", ".join(self.unique_key_columns)
        return (
            f"SELECT * FROM (\n{training_sql}\n) "
            f"WHERE MOD(ABS(HASH({key_args})), 100) < {buckets}"
        )

    # --- identity (design §3) -------------------------------------------
    def identity(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "base_table": self.base_table,
            "snowflake_database": self._database(),
            "snowflake_schema": self._schema(),
            "unique_key": list(self.unique_key_columns),
            "split_group_key": list(self.split_group_key_columns),
        }

    def recipe_identity(self, *, project_dir: str | Path | None = None) -> dict[str, Any]:
        # SQL files enter the recipe as content hashes, not paths: editing a
        # file is drift, renaming it is not (design §3).
        return {
            **self.identity(),
            "base_table_sql_sha256": _file_sha256(self.base_table_sql, project_dir),
            "training_data_sql_sha256": _file_sha256(self.training_data_sql, project_dir),
        }

    # --- trace (design §9) ------------------------------------------------
    def artifact_files(
        self,
        pipeline: "DataPipeline | None" = None,
        *,
        project_dir: str | Path | None = None,
    ) -> dict[str, Path]:
        # mkdtemp (not TemporaryDirectory): the returned paths must outlive
        # this call — _log_source_trace copies them into its own tempdir
        # before logging. One small leaked dir per real materialize; accepted.
        directory = Path(tempfile.mkdtemp(prefix="automl-snowflake-trace-"))
        files = {
            "base_table.executed.sql": self.generated_ddl(project_dir=project_dir),
            "training_data.executed.sql": self._render(self.training_data_sql, project_dir)
            .strip()
            .rstrip(";"),
        }
        out: dict[str, Path] = {}
        for name, text in files.items():
            path = directory / name
            path.write_text(text + "\n", encoding="utf-8")
            out[name] = path
        return out


def _scrub_sql(sql: str) -> str:
    """Drop string literals and ``--`` comments — for the guard checks only.

    The enforcement guards (single statement, SPLIT_PCT collision) must judge
    what the statement *does*, not what its comments or literal values say;
    the un-scrubbed body is what gets inserted into the DDL. Handles
    single-quoted literals with '' escapes; comments are stripped only
    outside literals.
    """
    out: list[str] = []
    i, n = 0, len(sql)
    in_literal = False
    while i < n:
        char = sql[i]
        if in_literal:
            if char == "'":
                if i + 1 < n and sql[i + 1] == "'":  # escaped quote inside literal
                    i += 2
                    continue
                in_literal = False
            i += 1
            continue
        if char == "'":
            in_literal = True
            i += 1
            continue
        if char == "-" and i + 1 < n and sql[i + 1] == "-":
            newline = sql.find("\n", i)
            i = n if newline == -1 else newline
            continue
        out.append(char)
        i += 1
    return "".join(out).strip()


def _file_sha256(sql_path: str | Path, project_dir: str | Path | None) -> str:
    path = Path(sql_path)
    resolved = path if path.is_absolute() else Path(project_dir or Path.cwd()) / path
    return "sha256:" + hashlib.sha256(resolved.read_bytes()).hexdigest()


__all__ = ["SnowflakeSource"]
