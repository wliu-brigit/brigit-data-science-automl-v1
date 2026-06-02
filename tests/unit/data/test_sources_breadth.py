import pytest

from automl.data import DataSpec, GCSParquetSource, SnowflakeSource
from automl.errors import StorageError

pytestmark = pytest.mark.unit


def test_gcs_parquet_source_is_public_and_has_deterministic_identity():
    source = GCSParquetSource(
        gcs_uri="gs://bucket/path/train.parquet",
        hash_key="row_id",
    )

    identity = source.identity()

    assert identity == {
        "kind": "gcs_parquet",
        "gcs_uri": "gs://bucket/path/train.parquet",
        "hash_key": ["row_id"],
    }
    assert DataSpec(source=source).source is source


def test_gcs_parquet_source_rejects_invalid_uri():
    with pytest.raises(ValueError, match="not-a-gcs-uri"):
        GCSParquetSource(gcs_uri="not-a-gcs-uri", hash_key="row_id")


def test_snowflake_source_is_public_and_has_stub_identity(monkeypatch):
    monkeypatch.setenv("SNOWFLAKE_DATABASE", "RAW_DB")
    monkeypatch.setenv("SNOWFLAKE_SCHEMA", "PUBLIC")

    source = SnowflakeSource(
        base_table="APP",
        base_data_sql="sql/base.sql",
        training_data_sql="sql/train.sql",
    )

    identity = source.identity()

    assert identity == {
        "kind": "snowflake",
        "base_table": "APP",
        "base_data_sql": "sql/base.sql",
        "training_data_sql": "sql/train.sql",
        "snowflake_database": "RAW_DB",
        "snowflake_schema": "PUBLIC",
        "hash_key": [],
    }
    assert DataSpec(source=source).source is source


def test_snowflake_source_load_fails_clearly_in_phase_two_stub():
    source = SnowflakeSource(
        base_table="APP",
        base_data_sql="sql/base.sql",
        training_data_sql="sql/train.sql",
    )

    with pytest.raises((NotImplementedError, StorageError), match="SnowflakeSource"):
        source.load()
