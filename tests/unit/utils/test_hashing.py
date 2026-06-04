import pandas as pd
import pytest

from automl.utils.hashing import dataframe_content_hash, json_hash, schema_hash

pytestmark = pytest.mark.unit


def test_json_hash_is_canonical_sha256_for_json_serializable_values():
    left = {"b": [2, 1], "a": {"nested": True}}
    right = {"a": {"nested": True}, "b": [2, 1]}

    assert json_hash(left) == json_hash(right)
    assert json_hash(left) == (
        "sha256:5f1247a96122f29c45132701341977d9f14c20e33fa300c92e5e41914f3f8038"
    )


def test_json_hash_rejects_non_json_serializable_values():
    with pytest.raises(TypeError):
        json_hash({"not_json": object()})


def test_schema_hash_is_sensitive_to_column_order_and_dtype_strings():
    ints = pd.DataFrame({"a": pd.Series([1], dtype="int64"), "b": ["x"]})
    reordered = ints[["b", "a"]]
    floats = pd.DataFrame({"a": pd.Series([1.0], dtype="float64"), "b": ["x"]})

    assert schema_hash(ints) != schema_hash(reordered)
    assert schema_hash(ints) != schema_hash(floats)
    assert schema_hash(ints) == (
        "sha256:8143b7b53dbafdea2e6654fa04f7bdc67cb81eb12127dec3bac98a810bc152ed"
    )


def test_dataframe_content_hash_is_sensitive_to_rows_columns_and_dtypes():
    df = pd.DataFrame({"id": pd.Series([1, 2], dtype="int64"), "value": ["a", "b"]})
    reordered_columns = df[["value", "id"]]
    changed_rows = pd.DataFrame({"id": pd.Series([1, 3], dtype="int64"), "value": ["a", "b"]})
    changed_dtype = pd.DataFrame(
        {"id": pd.Series([1.0, 2.0], dtype="float64"), "value": ["a", "b"]}
    )

    assert dataframe_content_hash(df) != dataframe_content_hash(reordered_columns)
    assert dataframe_content_hash(df) != dataframe_content_hash(changed_rows)
    assert dataframe_content_hash(df) != dataframe_content_hash(changed_dtype)


def test_content_hash_is_row_order_insensitive():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    shuffled = df.sample(frac=1, random_state=7).reset_index(drop=True)
    assert dataframe_content_hash(df) == dataframe_content_hash(shuffled)


def test_content_hash_still_counts_duplicates():
    once = pd.DataFrame({"a": [1, 2]})
    twice = pd.DataFrame({"a": [1, 1, 2]})
    assert dataframe_content_hash(once) != dataframe_content_hash(twice)


def test_content_hash_still_sees_column_order_and_dtypes():
    df = pd.DataFrame({"a": [1], "b": [2]})
    assert dataframe_content_hash(df) != dataframe_content_hash(df[["b", "a"]])
