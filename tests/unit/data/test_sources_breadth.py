import pytest

from automl.data import DataSpec, GCSParquetSource, LocalCSVSource, SnowflakeSource

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


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(
            lambda key: LocalCSVSource(csv_path="x.csv", unique_key=key), id="local_csv"
        ),
        pytest.param(
            lambda key: GCSParquetSource(gcs_uri="gs://b/p.parquet", unique_key=key),
            id="gcs_parquet",
        ),
        pytest.param(
            lambda key: SnowflakeSource(
                base_table="T",
                base_table_sql="a.sql",
                training_data_sql="b.sql",
                unique_key=key,
            ),
            id="snowflake",
        ),
    ],
)
def test_every_source_validates_keys_at_construction(make):
    # The DataSource base __post_init__ hook is the construction-edge
    # guarantee; this pins it firing for every source (a subclass override
    # that forgets super().__post_init__() would regress silently otherwise).
    with pytest.raises(ValueError, match="non-empty"):
        make(("a", "  "))
    with pytest.raises(ValueError, match="duplicate"):
        make(("a", "a"))
