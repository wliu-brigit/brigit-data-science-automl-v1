"""SnowflakeSource SQL assembly: rendering, DDL injection, enforcement."""

from pathlib import Path

import pandas as pd
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


def test_select_led_multi_statement_body_is_rejected(project):
    path = project / "data" / "queries" / "base_table.sql"
    path.write_text("SELECT 1; DROP TABLE T", encoding="utf-8")
    with pytest.raises(DataError, match="single SELECT"):
        _source().generated_ddl(project_dir=project)


def test_base_table_sql_emitting_split_pct_is_a_collision_error(project):
    path = project / "data" / "queries" / "base_table.sql"
    path.write_text("SELECT 1 AS SPLIT_PCT", encoding="utf-8")
    with pytest.raises(DataError, match="SPLIT_PCT"):
        _source().generated_ddl(project_dir=project)


def test_guards_ignore_inline_comments_and_string_literals(project):
    # The enforcement guards must not false-positive on comment text or
    # literal contents — only on what the statement actually does.
    path = project / "data" / "queries" / "base_table.sql"
    path.write_text(
        "SELECT t.TRANSACTION_ID, -- note; SPLIT_PCT used to be emitted here\n"
        "       'a;b' AS tag, 'SPLIT_PCT' AS label\n"
        "FROM {database}.{schema}.RAW_TXNS t",
        encoding="utf-8",
    )
    ddl = _source().generated_ddl(project_dir=project)
    assert "'a;b' AS tag" in ddl  # the body itself is inserted verbatim


def test_guards_still_catch_split_pct_behind_a_literal(project):
    # Scrubbing literals must not blind the guard to a real emission.
    path = project / "data" / "queries" / "base_table.sql"
    path.write_text("SELECT 'x' AS tag, 1 AS SPLIT_PCT", encoding="utf-8")
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
    renamed_path = project / "data" / "queries" / "base_table_renamed.sql"
    renamed_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    renamed = _source(base_table_sql="data/queries/base_table_renamed.sql")
    assert renamed.recipe_identity(project_dir=project) == after  # renaming isn't drift


def test_artifact_files_return_rendered_sql(project):
    files = _source().artifact_files(pipeline=None, project_dir=project)
    assert sorted(files) == ["base_table.executed.sql", "training_data.executed.sql"]
    assert "ML_DB.FRAUD" in Path(files["training_data.executed.sql"]).read_text(encoding="utf-8")


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


def test_table_exists_check_is_case_insensitive(project, fake_seam, monkeypatch):
    # Snowflake folds unquoted identifiers to UPPERCASE in INFORMATION_SCHEMA;
    # a lowercase SNOWFLAKE_SCHEMA in .env must not make the exists-check miss
    # (which would silently rebuild the base table on every load).
    monkeypatch.setenv("SNOWFLAKE_SCHEMA", "fraud")
    _source().load(project_dir=project)
    exists_sql = next(sql for sql in fake_seam["fetched"] if "INFORMATION_SCHEMA" in sql)
    assert "UPPER(TABLE_SCHEMA) = UPPER('fraud')" in exists_sql
    assert "UPPER(TABLE_NAME) = UPPER('FRAUD_TRAINING_BASE')" in exists_sql
    assert fake_seam["executed"] == []          # still no rebuild


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
