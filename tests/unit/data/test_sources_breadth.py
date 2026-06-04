import pytest

from automl.data import DataSpec, GCSParquetSource, LocalCSVSource, SnowflakeSource
from automl.errors import StorageError

pytestmark = pytest.mark.unit


def test_gcs_parquet_source_is_public_and_has_deterministic_identity():
    source = GCSParquetSource(
        gcs_uri="gs://bucket/path/train.parquet",
        unique_key="row_id",
    )

    identity = source.identity()

    assert identity == {
        "kind": "gcs_parquet",
        "gcs_uri": "gs://bucket/path/train.parquet",
        "unique_key": ["row_id"],
        "split_group_key": ["row_id"],
    }
    assert DataSpec(source=source).source is source


def test_gcs_parquet_source_rejects_invalid_uri():
    with pytest.raises(ValueError, match="not-a-gcs-uri"):
        GCSParquetSource(gcs_uri="not-a-gcs-uri", unique_key="row_id")


def test_split_group_key_defaults_to_unique_key_and_overrides():
    default = LocalCSVSource(csv_path="x.csv", unique_key="TXN_ID")
    assert default.split_group_key_columns == ("TXN_ID",)
    grouped = LocalCSVSource(csv_path="x.csv", unique_key="TXN_ID", split_group_key="USER_ID")
    assert grouped.unique_key_columns == ("TXN_ID",)
    assert grouped.split_group_key_columns == ("USER_ID",)


def test_sources_require_unique_key():
    with pytest.raises(TypeError):
        LocalCSVSource(csv_path="x.csv")  # unique_key is required


def test_snowflake_source_is_public_and_has_stub_identity(monkeypatch):
    monkeypatch.setenv("SNOWFLAKE_DATABASE", "RAW_DB")
    monkeypatch.setenv("SNOWFLAKE_SCHEMA", "PUBLIC")

    source = SnowflakeSource(
        base_table="APP",
        base_data_sql="sql/base.sql",
        training_data_sql="sql/train.sql",
        unique_key="row_id",
    )

    identity = source.identity()

    assert identity == {
        "kind": "snowflake",
        "base_table": "APP",
        "base_data_sql": "sql/base.sql",
        "training_data_sql": "sql/train.sql",
        "snowflake_database": "RAW_DB",
        "snowflake_schema": "PUBLIC",
        "unique_key": ["row_id"],
        "split_group_key": ["row_id"],
    }
    assert DataSpec(source=source).source is source


def test_snowflake_source_load_fails_clearly_in_phase_two_stub():
    source = SnowflakeSource(
        base_table="APP",
        base_data_sql="sql/base.sql",
        training_data_sql="sql/train.sql",
        unique_key="row_id",
    )

    with pytest.raises((NotImplementedError, StorageError), match="SnowflakeSource"):
        source.load()
