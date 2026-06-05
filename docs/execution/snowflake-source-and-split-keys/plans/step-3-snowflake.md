# Step 3 — SnowflakeSource

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A real `SnowflakeSource`: the `utils/io/snowflake.py` seam on
`snowflake-connector-python`, SELECT-only `base_table_sql` with
harness-owned DDL + `SPLIT_PCT` injection, the empirical split-invariant
check, bootstrap/rebuild inside `load()`, deterministic bucket-sample
dry-run, executed-SQL trace, a live `validate project` probe, new scaffold
templates, and Snowpark out of the lockfile.

**Architecture:** One seam file owns the connector (sibling of `gcs.py`);
the source is pure SQL-text assembly + seam calls, so every behavior is
unit-testable with the seam monkeypatched. Layer 1 verbs (bootstrap,
rebuild, invariant check) live *inside* `load()` — no new hooks (design §4).
Three green commits: (1) seam, (2) source + pipeline adoption, (3)
validation/scaffold/docs/deps.

**Tech stack:** `snowflake-connector-python[pandas]` (Arrow → pandas via
`fetch_pandas_all`), env-driven credentials from `.env` (loaded by the
existing project-config `load_dotenv`; **never handle credential values**).
The connector is already a **main project dependency** (`pyproject.toml:36`,
`snowflake-connector-python[pandas]>=3.0`) — verify before Task 1, and if it
were ever missing, add it with
`uv add "snowflake-connector-python[pandas]>=3.0"` (project-level, **not**
`--dev`: the seam is runtime library code, not tooling). Everything through
`uv`, never pip.

**Source of truth:** `../design.md` §9 (contract), §10 (connector), §11
(validation), §4 steps 2a–2d, §14 step 3.

**Prereqs:** Steps 1–2 landed. **No live Snowflake access is required to
land this step**: every behavior is unit-tested with the seam mocked, and
the green gate is `tests/unit tests/contracts tests/integration`. The e2e
test is *written* in Task 9 but **running it live is deferred to the
tail-end pass after step 4** (see the ledger's tail-end activities) — gated
on `AUTOML_E2E=1` + `SNOWFLAKE_*`, it skips cleanly until then. E2E work
uses a throwaway `dev_`-prefixed project (gitignored);
`fraud_anomaly_detection/` is touched **only** for the scaffold-contract
updates listed in Task 8; pre-existing warehouse tables are never replaced
without the explicit flag.

**Carried in from the step-1 review session (2026-06-04):** when rewriting
`SnowflakeSource`, give it the construction-time key validation the file
sources have — and prefer enforcing it once in the `DataSource` base (a
base-class hook) over copy-pasting the `__post_init__` property-touch idiom
a third time, so every future source inherits the construction-edge
guarantee. Also reconcile in Task 4 (pipeline adopts source-provided
SPLIT_PCT): `add_split_pct`'s internal collision guard checks the uppercase
column name against a frame `standardize_columns` has lowercased — dead on
the pipeline path; pick one altitude for the collision check while this
area is being rewritten.

---

## PART 1 — the seam (commit 1)

### Task 1: `automl/utils/io/snowflake.py` (TDD)

**Files:**
- Create: `tests/unit/utils/test_snowflake_io.py`
- Create: `automl/utils/io/snowflake.py`

- [ ] **Step 1: Write the failing tests** (mock the connector by injecting a
fake `snowflake.connector` module into `sys.modules` — the seam imports it
lazily inside functions, mirroring `gcs.py`'s lazy client import):

```python
"""Snowflake IO seam: env params, connection, fetch/execute."""

import sys
import types

import pandas as pd
import pytest

from automl.utils.io import snowflake as sf

pytestmark = pytest.mark.unit


@pytest.fixture
def fake_connector(monkeypatch):
    calls = {"connect_kwargs": None, "executed": [], "closed": False}

    class FakeCursor:
        def execute(self, sql):
            calls["executed"].append(sql)
            return self

        def fetch_pandas_all(self):
            return pd.DataFrame({"A": [1]})

        def fetchone(self):
            return (1,)

        def close(self):
            pass

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

        def close(self):
            calls["closed"] = True

    def connect(**kwargs):
        calls["connect_kwargs"] = kwargs
        return FakeConnection()

    fake = types.ModuleType("snowflake.connector")
    fake.connect = connect
    monkeypatch.setitem(sys.modules, "snowflake", types.ModuleType("snowflake"))
    monkeypatch.setitem(sys.modules, "snowflake.connector", fake)
    return calls


def _set_env(monkeypatch, **extra):
    for name in ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD"):
        monkeypatch.setenv(name, "x")
    for name, value in extra.items():
        monkeypatch.setenv(name, value)


def test_missing_env_lists_exactly_whats_missing(monkeypatch):
    for name in ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SNOWFLAKE_USER", "u")
    assert sf.missing_env() == ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_PASSWORD"]


def test_connection_params_apply_warehouse_and_role_defaults(monkeypatch, fake_connector):
    _set_env(monkeypatch)
    monkeypatch.delenv("SNOWFLAKE_WAREHOUSE", raising=False)
    monkeypatch.delenv("SNOWFLAKE_ROLE", raising=False)
    sf.execute("SELECT 1")
    assert fake_connector["connect_kwargs"]["warehouse"] == "DATA_SCIENCE_WH"
    assert fake_connector["connect_kwargs"]["role"] == "DATA_SCIENCE_ROLE"
    assert fake_connector["closed"] is True


def test_fetch_df_returns_pandas(monkeypatch, fake_connector):
    _set_env(monkeypatch)
    out = sf.fetch_df("SELECT A FROM T")
    assert list(out.columns) == ["A"]
    assert fake_connector["executed"] == ["SELECT A FROM T"]


def test_check_connection_runs_select_1(monkeypatch, fake_connector):
    _set_env(monkeypatch)
    sf.check_connection()
    assert fake_connector["executed"] == ["SELECT 1"]


def test_missing_env_raises_before_any_connection(monkeypatch):
    monkeypatch.delenv("SNOWFLAKE_ACCOUNT", raising=False)
    with pytest.raises(EnvironmentError, match="SNOWFLAKE_ACCOUNT"):
        sf.execute("SELECT 1")
```

- [ ] **Step 2:** `uv run pytest tests/unit/utils/test_snowflake_io.py -v` —
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `automl/utils/io/snowflake.py`**

```python
"""Snowflake connection seam (sibling of gcs.py).

The single place that talks to the warehouse. Credentials come from the
environment only (.env via the project-config load_dotenv); this module
never logs or returns credential values. The connector choice
(snowflake-connector-python; Snowpark deliberately dropped — design §10)
is encapsulated here: if recipes ever need warehouse-side Python, swapping
is a one-file change.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

import pandas as pd


REQUIRED_ENV = ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD")
DEFAULT_WAREHOUSE = "DATA_SCIENCE_WH"
DEFAULT_ROLE = "DATA_SCIENCE_ROLE"


def missing_env() -> list[str]:
    """Names of required env vars that are unset/empty (for validate + errors)."""
    return [name for name in REQUIRED_ENV if not os.environ.get(name)]


def connection_params() -> dict[str, Any]:
    missing = missing_env()
    if missing:
        raise EnvironmentError(
            f"Missing required environment variable(s): {', '.join(missing)} "
            "(set them in the repo-root .env)"
        )
    return {
        "account": os.environ["SNOWFLAKE_ACCOUNT"],
        "user": os.environ["SNOWFLAKE_USER"],
        "password": os.environ["SNOWFLAKE_PASSWORD"],
        "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE") or DEFAULT_WAREHOUSE,
        "role": os.environ.get("SNOWFLAKE_ROLE") or DEFAULT_ROLE,
    }


@contextmanager
def connect() -> Iterator[Any]:
    import snowflake.connector

    connection = snowflake.connector.connect(**connection_params())
    try:
        yield connection
    finally:
        connection.close()


def fetch_df(sql: str) -> pd.DataFrame:
    """Run a SELECT and return the result as pandas (Arrow-backed fetch)."""
    with connect() as connection:
        cursor = connection.cursor()
        try:
            cursor.execute(sql)
            return cursor.fetch_pandas_all()
        finally:
            cursor.close()


def fetch_one(sql: str) -> tuple | None:
    with connect() as connection:
        cursor = connection.cursor()
        try:
            cursor.execute(sql)
            return cursor.fetchone()
        finally:
            cursor.close()


def execute(sql: str) -> None:
    with connect() as connection:
        cursor = connection.cursor()
        try:
            cursor.execute(sql)
        finally:
            cursor.close()


def check_connection() -> None:
    """Cheapest possible live probe; driver errors propagate verbatim."""
    execute("SELECT 1")


__all__ = [
    "DEFAULT_ROLE",
    "DEFAULT_WAREHOUSE",
    "REQUIRED_ENV",
    "check_connection",
    "connect",
    "connection_params",
    "execute",
    "fetch_df",
    "fetch_one",
    "missing_env",
]
```

- [ ] **Step 4:** `uv run pytest tests/unit/utils/test_snowflake_io.py -v` — PASS.

- [ ] **Step 5: Commit**

```bash
git add automl/utils/io/snowflake.py tests/unit/utils/test_snowflake_io.py
git commit -m "Add the Snowflake IO seam on snowflake-connector-python (design step 3, part 1)"
```

---

## PART 2 — the real source + pipeline adoption (commit 2)

### Task 2: SQL assembly — rendering, DDL generation, enforcement (TDD)

**Files:**
- Create: `tests/unit/data/test_snowflake_source.py` (replaces the two stub
  tests — delete `test_snowflake_source_is_public_and_has_stub_identity` and
  `test_snowflake_source_load_fails_clearly_in_phase_two_stub` from
  `tests/unit/data/test_sources_breadth.py` in this task)
- Modify: `automl/data/sources/snowflake.py`

- [ ] **Step 1: Write the failing tests** (pure text in/out — no seam needed
for this task):

```python
"""SnowflakeSource SQL assembly: rendering, DDL injection, enforcement."""

from pathlib import Path

import pytest

from automl.data.sources.snowflake import SnowflakeSource
from automl.errors import DataError

pytestmark = pytest.mark.unit


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("SNOWFLAKE_DATABASE", "ML_DB")
    monkeypatch.setenv("SNOWFLAKE_SCHEMA", "FRAUD")
    queries = tmp_path / "data" / "queries"
    queries.mkdir(parents=True)
    (queries / "base_table.sql").write_text(
        "-- base data\nSELECT t.TRANSACTION_ID, t.USER_ID, t.AMOUNT\n"
        "FROM {database}.{schema}.RAW_TXNS t\n",
        encoding="utf-8",
    )
    (queries / "training_data.sql").write_text(
        "SELECT * FROM {database}.{schema}.{base_table}\n", encoding="utf-8"
    )
    return tmp_path


def _source(**overrides):
    kwargs = dict(
        base_table="FRAUD_TRAINING_BASE",
        base_table_sql="data/queries/base_table.sql",
        training_data_sql="data/queries/training_data.sql",
        unique_key="TRANSACTION_ID",
        split_group_key="USER_ID",
    )
    kwargs.update(overrides)
    return SnowflakeSource(**kwargs)


def test_generated_ddl_wraps_select_and_injects_split_pct(project):
    ddl = _source().generated_ddl(project_dir=project)
    assert ddl.startswith("CREATE OR REPLACE TABLE ML_DB.FRAUD.FRAUD_TRAINING_BASE AS")
    assert "MOD(ABS(HASH(t.USER_ID)), 100) AS SPLIT_PCT" in ddl
    assert "FROM {database}" not in ddl  # substitutions applied
    assert "ML_DB.FRAUD.RAW_TXNS" in ddl


def test_generated_ddl_supports_composite_split_group_key(project):
    source = _source(split_group_key=("USER_ID", "MERCHANT_ID"))
    assert "HASH(t.MERCHANT_ID, t.USER_ID)" in source.generated_ddl(project_dir=project)


def test_base_table_sql_must_be_a_single_select(project):
    path = project / "data" / "queries" / "base_table.sql"
    path.write_text("DELETE FROM T; SELECT 1", encoding="utf-8")
    with pytest.raises(DataError, match="single SELECT"):
        _source().generated_ddl(project_dir=project)


def test_base_table_sql_emitting_split_pct_is_a_collision_error(project):
    path = project / "data" / "queries" / "base_table.sql"
    path.write_text("SELECT 1 AS SPLIT_PCT", encoding="utf-8")
    with pytest.raises(DataError, match="SPLIT_PCT"):
        _source().generated_ddl(project_dir=project)


def test_with_clause_is_accepted(project):
    path = project / "data" / "queries" / "base_table.sql"
    path.write_text("WITH t AS (SELECT 1 AS TRANSACTION_ID) SELECT * FROM t", encoding="utf-8")
    assert _source().generated_ddl(project_dir=project)


def test_identifiers_are_not_case_mangled(project, monkeypatch):
    monkeypatch.setenv("SNOWFLAKE_SCHEMA", "Fraud_Mixed")
    ddl = _source().generated_ddl(project_dir=project)
    assert "ML_DB.Fraud_Mixed.FRAUD_TRAINING_BASE" in ddl  # old impl lowercased; dropped


def test_recipe_identity_hashes_sql_content_not_paths(project):
    before = _source().recipe_identity(project_dir=project)
    path = project / "data" / "queries" / "base_table.sql"
    path.write_text(path.read_text(encoding="utf-8") + "\n-- edited", encoding="utf-8")
    after = _source().recipe_identity(project_dir=project)
    assert before["base_table_sql_sha256"] != after["base_table_sql_sha256"]
    assert "base_table_sql" not in before  # paths are not identity (renames aren't drift)
    renamed = _source(base_table_sql="data/queries/base_table.sql")
    assert set(before) == set(renamed.recipe_identity(project_dir=project))


def test_artifact_files_return_rendered_sql(project):
    files = _source().artifact_files(pipeline=None, project_dir=project)
    assert sorted(files) == ["base_table.executed.sql", "training_data.executed.sql"]
    assert "ML_DB.FRAUD" in Path(files["training_data.executed.sql"]).read_text(encoding="utf-8")
```

- [ ] **Step 2:** Run — FAIL (`generated_ddl` missing; `base_table_sql`
field missing).

- [ ] **Step 3: Implement the assembly half of `automl/data/sources/snowflake.py`**

```python
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
    base_table_sql: str | Path        # the SELECT defining the base data (renamed from base_data_sql)
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
        # Explicit replace, NOT str.format (review finding): Snowflake SQL
        # legitimately contains literal braces (OBJECT_CONSTRUCT('{...}'),
        # semi-structured paths) which str.format turns into an opaque
        # KeyError. Only the three documented substitutions are special.
        # No case-mangling of identifiers (the old implementation lowercased
        # env values; dropped as surprising — design §9).
        return (
            text.replace("{database}", self._database())
            .replace("{schema}", self._schema())
            .replace("{base_table}", self.base_table)
        )

    # --- DDL generation (design §9) -------------------------------------
    def generated_ddl(self, *, project_dir: str | Path | None = None) -> str:
        body = self._render(self.base_table_sql, project_dir).strip().rstrip(";").strip()
        statement = _strip_sql_comments(body)
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


def _strip_sql_comments(sql: str) -> str:
    lines = [line for line in sql.splitlines() if not line.lstrip().startswith("--")]
    return "\n".join(lines).strip()


def _file_sha256(sql_path: str | Path, project_dir: str | Path | None) -> str:
    path = Path(sql_path)
    resolved = path if path.is_absolute() else Path(project_dir or Path.cwd()) / path
    return "sha256:" + hashlib.sha256(resolved.read_bytes()).hexdigest()


__all__ = ["SnowflakeSource"]
```

**Interface note:** `artifact_files` gains a keyword `project_dir`. Three
places change (review found the third): the base
`DataSource.artifact_files(pipeline)` signature gains the optional keyword
(default `None`, ignored by other sources); the call site in
`automl/data/pipeline.py` `_materialize_bound` passes
`project_dir=active.config.project_dir`; and the **fake source in
`tests/integration/data_pipeline/test_materialize_load.py:242`** must become
`def artifact_files(self, pipeline=None, *, project_dir=None)` or it
TypeErrors on the new kwarg.

- [ ] **Step 4:** `uv run pytest tests/unit/data/test_snowflake_source.py -v` — PASS.

### Task 3: `load()` — bootstrap, rebuild, invariant, pull, dry-run (TDD)

**Files:**
- Modify: `tests/unit/data/test_snowflake_source.py`
- Modify: `automl/data/sources/snowflake.py`

- [ ] **Step 1: Write the failing tests** (monkeypatch the seam functions on
`automl.data.sources.snowflake.sf` — the source's only outward calls):

```python
@pytest.fixture
def fake_seam(monkeypatch):
    state = {
        "table_exists": True,
        "invariant_violations": 0,
        "total_rows": 1_000,
        "executed": [],
        "fetched": [],
    }

    def fetch_one(sql):
        state["fetched"].append(sql)
        if "INFORMATION_SCHEMA.TABLES" in sql:
            return (1,) if state["table_exists"] else None
        if "IS DISTINCT FROM" in sql:
            return (1,) if state["invariant_violations"] else None
        if "COUNT(*)" in sql:
            return (state["total_rows"],)
        raise AssertionError(f"unexpected fetch_one: {sql}")

    def execute(sql):
        state["executed"].append(sql)
        if sql.startswith("CREATE OR REPLACE TABLE"):
            state["table_exists"] = True
            state["invariant_violations"] = 0

    def fetch_df(sql):
        state["fetched"].append(sql)
        return pd.DataFrame(
            {"TRANSACTION_ID": [1, 2], "USER_ID": ["a", "b"], "SPLIT_PCT": [3, 97]}
        )

    monkeypatch.setattr("automl.data.sources.snowflake.sf.fetch_one", fetch_one)
    monkeypatch.setattr("automl.data.sources.snowflake.sf.execute", execute)
    monkeypatch.setattr("automl.data.sources.snowflake.sf.fetch_df", fetch_df)
    return state


def test_load_pulls_without_ddl_when_table_exists(project, fake_seam):
    out = _source().load(project_dir=project)
    assert fake_seam["executed"] == []          # no rebuild
    assert "SPLIT_PCT" in out.columns


def test_load_bootstraps_when_table_missing(project, fake_seam):
    fake_seam["table_exists"] = False
    _source().load(project_dir=project)
    assert any(sql.startswith("CREATE OR REPLACE TABLE") for sql in fake_seam["executed"])


def test_refresh_source_rebuilds_even_when_table_exists(project, fake_seam):
    _source().load(project_dir=project, refresh_source=True)
    assert any(sql.startswith("CREATE OR REPLACE TABLE") for sql in fake_seam["executed"])


def test_invariant_mismatch_errors_naming_refresh_source(project, fake_seam):
    fake_seam["invariant_violations"] = 1
    with pytest.raises(DataError, match="--refresh-source"):
        _source().load(project_dir=project)
    assert fake_seam["executed"] == []          # never auto-spends warehouse minutes


def test_dry_run_wraps_with_deterministic_bucket_filter(project, fake_seam):
    fake_seam["total_rows"] = 1_000
    _source().load(project_dir=project, nrows=100)   # 100/1000 -> k = 10
    pull = fake_seam["fetched"][-1]
    assert "WHERE SPLIT_PCT < 10" in pull
    # same sample every run: repeat and compare
    _source().load(project_dir=project, nrows=100)
    assert fake_seam["fetched"][-1] == pull


def test_dry_run_bucket_count_is_clamped_to_at_least_one(project, fake_seam):
    fake_seam["total_rows"] = 10_000_000
    _source().load(project_dir=project, nrows=100)
    assert "WHERE SPLIT_PCT < 1" in fake_seam["fetched"][-1]
```

- [ ] **Step 2:** Run — FAIL (`load` still the stub).

- [ ] **Step 3: Implement `load()` and its private steps** (replacing the
stub; design §4 steps 2a–2d live *inside* this one method):

```python
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
        row = sf.fetch_one(
            f"SELECT 1 FROM {self._database()}.INFORMATION_SCHEMA.TABLES "
            f"WHERE TABLE_SCHEMA = '{self._schema()}' AND TABLE_NAME = '{self.base_table}'"
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
        row = sf.fetch_one(f"SELECT COUNT(*) FROM {self._qualified_table()}")
        total = int(row[0]) if row and row[0] else 0
        if total <= 0:
            buckets = 100
        else:
            buckets = max(1, min(100, round(100 * nrows / total)))
        # Whole buckets trade exactness for determinism: the same sample every
        # run, so dry-run identity is stable (design §4).
        return (
            f"SELECT * FROM (\n{training_sql}\n) "
            f"WHERE {SPLIT_PCT_COL} < {buckets}"
        )
```

- [ ] **Step 4:** `uv run pytest tests/unit/data/test_snowflake_source.py -v` — PASS.

### Task 4: pipeline adopts a source-provided SPLIT_PCT

**Files:**
- Modify: `automl/data/sources/base.py`
- Modify: `automl/data/pipeline.py`
- Modify: `tests/unit/data/test_sources_pipeline_contract.py`

- [ ] **Step 1:** `DataSource` gains the class attribute (default for file
sources):

```python
class DataSource(ABC):
    kind = "base"
    provides_split_pct = False   # True when SPLIT_PCT arrives from the source (Snowflake)
```

- [ ] **Step 2:** In `DataPipeline.run()`, replace the unconditional
collision-check + assignment block with the branch (this also fixes the
old lowercased-`splitid`-feature-column leakage bug — design §8):

```python
        if self.spec.source.provides_split_pct:
            df = self._adopt_provided_split_pct(df, original_names)
        else:
            self._check_split_pct_collision(original_names)
            df = add_split_pct(
                df, split_group_key=split_group_key, split_pct_col=self.split_pct_col
            )
        validate_unique_key(df, unique_key=unique_key, source_label=self.spec.source.kind)
        validate_split_pct(df, split_pct_col=self.split_pct_col, source_label=self.spec.source.kind)
```

with:

```python
    def _adopt_provided_split_pct(
        self, df: pd.DataFrame, original_names: dict[str, str]
    ) -> pd.DataFrame:
        """Restore the source-provided split column to its canonical name.

        Column standardization lowercases SPLIT_PCT like any other column;
        for sources that own bucket assignment it is a protected pipeline
        column, never a feature — so it is renamed back, not recomputed.
        """
        lowered = self.split_pct_col.lower()
        if lowered not in df.columns:
            raise DataError(
                f"{self.spec.source.kind} declares provides_split_pct but no "
                f"{self.split_pct_col} column arrived; carry it through from the base table"
            )
        out = df.rename(columns={lowered: self.split_pct_col})
        original_names[self.split_pct_col] = original_names.pop(lowered, self.split_pct_col)
        return out
```

The registry build already treats `split_pct_col` as metadata, so the
restored column is excluded from features by construction. The record notes
provenance — in `_dataset_for`, add to `source_identity`:

```python
        source_identity["split"] = (
            "sql"
            if self.spec.source.provides_split_pct
            else f"python(split_group_key={list(split_group_key)})"
        )
```

- [ ] **Step 3: Add a pipeline-level test** in
`test_sources_pipeline_contract.py` — a fake `DataSource` subclass with
`provides_split_pct = True` whose `load()` returns a frame containing
`SPLIT_PCT`; assert `build_dataset` keeps the provided values verbatim
(no recompute), the column survives under its canonical name, the registry
marks it non-feature, and `source_identity["split"] == "sql"`. Plus the
missing-column case asserting the "carry it through" error.

- [ ] **Step 4:** `uv run pytest tests/unit/data tests/integration/data_pipeline -v` — PASS.

- [ ] **Step 5: Commit**

```bash
git add -A automl tests
git commit -m "Real SnowflakeSource: harness-owned DDL with SPLIT_PCT injection, invariant check, bucket-sample dry-run

base_table_sql is a SELECT; bootstrap/rebuild live inside load(); the
pipeline adopts source-provided SPLIT_PCT instead of recomputing
(design step 3, part 2)."
```

---

## PART 3 — validation, scaffold, docs, deps (commit 3)

### Task 5: live `project.connections.snowflake` probe

**Files:**
- Modify: `automl/project/checks.py`
- Modify: `tests/unit/project/test_project_validation.py`

- [ ] **Step 1: Replace `snowflake_connection()`** (today: the pending
warning at `checks.py:166-178`) with the live probe. **Signature
(review-verified):** check functions take `(*, config)` and are invoked via
`run_check("project.connections.snowflake", snowflake_connection,
config=config)` at `checks.py:59-65` — a `session` parameter would
`TypeError` at runtime. Mirror the GCS/MLflow neighbors exactly (including
their `# noqa: BLE001` broad-except idiom):

```python
def snowflake_connection(*, config: Any) -> Iterable[Issue]:
    spec = getattr(config, "data_spec", None)
    source = getattr(spec, "source", None)
    if getattr(source, "kind", "") != "snowflake":
        return
    from automl.utils.io import snowflake as sf

    missing = sf.missing_env()
    if missing:
        yield Issue(
            level="error",
            check="project.connections.snowflake",
            message=f"missing Snowflake environment variable(s): {', '.join(missing)}",
        )
        return
    for label, sql_path in (
        ("base_table_sql", source.base_table_sql),
        ("training_data_sql", source.training_data_sql),
    ):
        path = Path(sql_path)
        resolved = path if path.is_absolute() else config.project_dir / path
        if not resolved.exists():
            yield Issue(
                level="error",
                check="project.connections.snowflake",
                message=f"{label} file not found: {resolved}",
            )
    try:
        sf.check_connection()
    except Exception as exc:  # noqa: BLE001 - driver errors surface verbatim
        yield Issue(
            level="error",
            check="project.connections.snowflake",
            message=f"Snowflake connection failed: {exc}",
        )
```

(Follow `placeholder_values` at `checks.py:101-116` for the
config-attribute resolution pattern.)

- [ ] **Step 2: Replace the pinned test** —
`test_validate_project_live_marks_snowflake_pending` (asserts `"pending"`,
warning level) becomes three tests: missing env → error listing the names;
env present + connection mocked OK + SQL files present → no snowflake issue;
connection raising → error containing the driver message verbatim. Use
`monkeypatch.setattr("automl.utils.io.snowflake.check_connection", ...)`.

- [ ] **Step 3:** `uv run pytest tests/unit/project -v` — PASS.

### Task 6: pending-language docs updated with the shape

**Files (review-expanded — the old `base_data` contract survives in more
places than the pending-language passages):**
- Modify: `agent-skills/skills/setup/SKILL.md` (lines ~155–156 pending
  language; line ~118 `base_data.sql` filename)
- Modify: `agent-skills/skills/validate/SKILL.md` (lines ~39–41)
- Modify: `agent-skills/references/setup/data-pipeline.md` (lines ~12, 56,
  85 `base_data_sql`)
- Modify: `agent-skills/references/setup/snowflake.md` (describes the old
  two-query contract end to end — rewrite to §9)
- Modify: `agent-skills/references/loop/leakage.md` (line ~11 `base_data`)
- Modify: `agent-skills/agents/automl-coder.md` (line ~40 `base_data`)

- [ ] **Step 1:** Replace the "pending source implementation / not checked
yet" passages: `project.connections.snowflake` is now a **live probe**
(missing env vars → error listing which; else `SELECT 1`, driver errors
verbatim; SQL files must exist) emitted only for Snowflake-backed projects.
Update the data-pipeline reference's SnowflakeSource section to the §9
contract: `base_table_sql` is a SELECT, the harness owns the CREATE and
injects `SPLIT_PCT` from `split_group_key`, `training_data_sql` pulls rows
and must carry `SPLIT_PCT` through, substitutions
`{database}`/`{schema}`/`{base_table}`, dry-run = deterministic bucket
sample.

- [ ] **Step 2:** Verify no stale phrase or filename survives (two gates —
the second catches contract drift the first can't):

```bash
grep -rn "pending source implementation\|not checked yet\|not implemented" agent-skills automl | grep -i snowflake
grep -rn "base_data" agent-skills automl tests projects | grep -v docs/archive
```

Expected: no output from either.

### Task 7: scaffold templates → the new contract

**Files:**
- Modify: `automl/project/scaffold.py`
- Modify: `tests/unit/project/test_metadata_and_scaffold.py`

- [ ] **Step 1:** In `_CONFIG_TEMPLATE`, the source block becomes the §9
contract (note the field rename `base_data_sql` → `base_table_sql` and file
rename `base_data.sql` → `base_table.sql`):

```python
source = SnowflakeSource(
    base_table="<TBD_base_table>",  # name only; lands at {database}.{schema}.{base_table}
    base_table_sql="data/queries/base_table.sql",      # the SELECT defining the base data
    training_data_sql="data/queries/training_data.sql",  # the SELECT pulling training rows
    unique_key="<TBD_unique_key>",  # stable row identifier; tuple for composite keys
    # split_group_key="USER_ID",    # declare only when splits must group by a coarser key
)
```

- [ ] **Step 2:** `_snowflake_templates()` emits the new starters:

`base_table.sql`:

```sql
-- The SELECT that defines your base data: joins, CTEs, filters, feature SQL.
-- The harness wraps it in CREATE OR REPLACE TABLE and injects SPLIT_PCT from
-- split_group_key — do not emit SPLIT_PCT yourself.
SELECT *
FROM {database}.{schema}.<TBD_SOURCE_TABLE>
```

`training_data.sql`:

```sql
-- The SELECT that pulls training rows from the base table.
-- SPLIT_PCT flows through; keep it in the projection.
SELECT *
FROM {database}.{schema}.{base_table}
```

- [ ] **Step 3:** Update `test_metadata_and_scaffold.py` — three distinct
edits (review-corrected):
  - `CONFIG_PLACEHOLDERS` stays **config-only** — `("<TBD_target_column>",
    "<TBD_base_table>", "<TBD_unique_key>", "TBD_experiment_id")`. Do NOT
    add `<TBD_SOURCE_TABLE>`: the test asserts every entry is `in` the
    config.py text, and that placeholder lives only in `base_table.sql` —
    assert it separately against the scaffolded SQL file.
  - The hard filename assertion at **line 62** (`data/queries/base_data.sql`
    exists) → `base_table.sql`.
  - The `<TBD_SPLIT_GROUP_KEY_COLUMN>` placeholder from step 1 is gone —
    the user no longer writes the hash line.

### Task 8: fraud project onto the new contract (explicitly required)

**Files:**
- Modify: `projects/fraud_anomaly_detection/config.py`
- Rename: `projects/fraud_anomaly_detection/data/queries/base_data.sql` →
  `base_table.sql`
- Modify: both SQL files

- [ ] **Step 1:** `config.py`: `base_data_sql="data/queries/base_data.sql"` →
`base_table_sql="data/queries/base_table.sql"` (keys already present from
step 1). Replace the two SQL files with the Task 7 starter contents (they
are still all-placeholder; the old `CREATE OR REPLACE` starter no longer
matches the contract). `git mv` for the rename. Nothing else in the project
changes.

### Task 9: drop Snowpark; e2e test; commit 3

**Files:**
- Modify: `pyproject.toml` (line 37)
- Create: `tests/e2e/test_snowflake_source_e2e.py`
- Check: `tests/contracts/test_environment.py`

- [ ] **Step 1:** Delete `"snowflake-snowpark-python>=1.50.0"` from
`pyproject.toml`; run `uv sync`; confirm
`uv run python -c "import snowflake.connector; print('ok')"` still prints
`ok` (the connector is a direct dependency, line 36).

- [ ] **Step 2:** `grep -rn "snowpark" automl tests pyproject.toml uv.lock | grep -v "^uv.lock"`
→ no source hits. If `tests/contracts/test_environment.py` pins the
dependency list, update it in the same change.

- [ ] **Step 3: E2E test** — gated like the rest of
`tests/e2e/_gates.py`, additionally requiring the Snowflake env:

```python
"""Live SnowflakeSource materialize, against a throwaway dev_ project."""

import os

import pytest

from tests.e2e._gates import LIVE_E2E_ENV, SERVICE_ENV

SNOWFLAKE_ENV = (
    "SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD",
    "SNOWFLAKE_DATABASE", "SNOWFLAKE_SCHEMA",
)
_required = (LIVE_E2E_ENV, *SERVICE_ENV, *SNOWFLAKE_ENV)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        any(not os.environ.get(name) for name in _required),
        reason=f"snowflake e2e requires {', '.join(_required)}",
    ),
]


def test_materialize_bootstraps_pulls_and_attaches(tmp_path):
    # Build a dev_-prefixed throwaway project dir (config with SnowflakeSource
    # over a tiny SELECT against a known-small table), then:
    #   1. materialize() -> bootstraps the base table, mints v1, SPLIT_PCT valid
    #   2. materialize() again -> attaches to v1 with zero warehouse queries
    #   3. materialize(refresh_data=True) -> content identity dedups to v1
    ...
```

Write the body against whatever `dev_` table wendao designates at execution
time — the table name comes from the env/conversation, never hardcoded to a
production table. **Write-only in this step: do not run it live.** The live
run requires `AUTOML_E2E=1` plus credentials in `.env` and is **deferred to
the tail-end pass after step 4** (ledger tail-end activities); in this
session just verify the gate makes it *skip*, not fail
(`uv run pytest tests/e2e/test_snowflake_source_e2e.py` → skipped).

- [ ] **Step 4:** Full suite: `uv run pytest tests/unit tests/contracts tests/integration` — PASS.

- [ ] **Step 5:** Update `docs/HANDOFF.md` (step 3 landed; next: step 4,
flexible splits; live fraud bootstrap can now be exercised).

- [ ] **Step 6: Commit**

```bash
git add -A automl tests projects agent-skills pyproject.toml uv.lock docs/HANDOFF.md
git commit -m "Live Snowflake validation probe, new scaffold contract, Snowpark dropped

project.connections.snowflake becomes a real SELECT 1 probe; scaffold and
fraud project move to base_table_sql-as-SELECT; snowflake-snowpark-python
leaves the lockfile (design step 3, part 3)."
```

---

## Self-review checklist

- [ ] The invariant check runs on every real pull and **before** any fetch;
  a mismatch never triggers an automatic rebuild (the error names the flag).
- [ ] Bootstrap happens only when the table is missing; `--refresh-source`
  is the only path that replaces an existing table (ground rule).
- [ ] No credential value ever appears in code, logs, errors, or tests.
- [ ] `base_data_sql` no longer exists anywhere:
  `grep -rn "base_data_sql\|base_data.sql" automl tests projects agent-skills | grep -v docs/archive` → empty.
- [ ] Composite `split_group_key` produces `HASH(t.A, t.B)` in sorted column
  order, matching `split_group_key_columns`.
- [ ] Stub-pinning tests are gone; `test_sources_breadth.py` keeps only the
  file-source surface.
- [ ] Recorded caveat (design §8) lands in the data-pipeline reference doc:
  Snowflake `HASH()` and pandas assign different buckets — CSV→Snowflake
  migrations reshuffle split membership.
