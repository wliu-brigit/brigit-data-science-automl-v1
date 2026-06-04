"""Key normalization, SPLIT_PCT assignment, and materialize-edge validation."""

import pandas as pd
import pytest

from automl.data.split import (
    SPLIT_PCT_COL,
    add_split_pct,
    split_report,
    validate_split_pct,
    validate_unique_key,
)
from automl.errors import DataError

pytestmark = pytest.mark.unit


def _frame():
    return pd.DataFrame({"user_id": ["a", "b", "c", "a"], "txn_id": [1, 2, 3, 4], "x": [0.1, 0.2, 0.3, 0.4]})


def test_add_split_pct_assigns_deterministic_buckets_from_group_key():
    out1 = add_split_pct(_frame(), split_group_key=("user_id",))
    out2 = add_split_pct(_frame(), split_group_key=("user_id",))
    assert SPLIT_PCT_COL in out1.columns
    assert out1[SPLIT_PCT_COL].between(0, 99).all()
    assert out1[SPLIT_PCT_COL].tolist() == out2[SPLIT_PCT_COL].tolist()
    # same group key value -> same bucket (rows 0 and 3 share user_id "a")
    assert out1[SPLIT_PCT_COL].iloc[0] == out1[SPLIT_PCT_COL].iloc[3]


def test_add_split_pct_errors_when_source_already_provides_the_column():
    df = _frame()
    df[SPLIT_PCT_COL] = 0
    with pytest.raises(DataError, match="SPLIT_PCT"):
        add_split_pct(df, split_group_key=("user_id",))


def test_add_split_pct_errors_on_missing_group_key_column():
    with pytest.raises(KeyError, match="split_group_key"):
        add_split_pct(_frame(), split_group_key=("nope",))


def test_validate_unique_key_passes_for_unique_tuples():
    validate_unique_key(_frame(), unique_key=("txn_id",))
    validate_unique_key(_frame(), unique_key=("txn_id", "user_id"))


def test_validate_unique_key_errors_on_duplicates_with_examples():
    with pytest.raises(DataError, match="duplicate"):
        validate_unique_key(_frame(), unique_key=("user_id",))


def test_validate_unique_key_errors_on_missing_columns():
    with pytest.raises(DataError, match="unique_key"):
        validate_unique_key(_frame(), unique_key=("nope",))


def test_validate_split_pct_accepts_integer_0_99():
    df = _frame()
    df[SPLIT_PCT_COL] = [0, 50, 99, 7]
    validate_split_pct(df)


@pytest.mark.parametrize(
    "values, match",
    [([0, 50, 99, 100], "0–99|0-99"), ([0.5, 1.0, 2.0, 3.0], "integer"), (None, "missing")],
)
def test_validate_split_pct_rejects_bad_columns(values, match):
    df = _frame()
    if values is not None:
        df[SPLIT_PCT_COL] = values
    with pytest.raises(DataError, match=match):
        validate_split_pct(df)


def test_validate_split_pct_rejects_nullable_integer_with_missing_values():
    df = _frame()
    df[SPLIT_PCT_COL] = pd.array([0, 50, 99, None], dtype="Int64")
    with pytest.raises(DataError, match="missing values"):
        validate_split_pct(df)


def test_validate_unique_key_rejects_null_key_values_even_without_duplicates():
    df = _frame().iloc[:3].copy()  # txn_id 1, 2, 3 — no duplicates
    df.loc[2, "txn_id"] = None
    with pytest.raises(DataError, match="null"):
        validate_unique_key(df, unique_key=("txn_id",))


def test_split_report_counts_buckets():
    df = add_split_pct(_frame(), split_group_key=("txn_id",))
    report = split_report(df)
    assert int(report["rows"].sum()) == len(df)


def test_normalize_key_sorts_and_rejects_bad_declarations():
    # _normalize_key is module-internal by settled decision (2026-06-04); these
    # tests pin its invariants because dataset identity depends on them:
    # sorted output keeps composite-key declaration order out of identity hashes.
    from automl.data.split import _normalize_key

    assert _normalize_key("TXN_ID", field_name="unique_key") == ("TXN_ID",)
    assert _normalize_key(("user_id", "txn_id"), field_name="unique_key") == ("txn_id", "user_id")
    assert _normalize_key(("txn_id", "user_id"), field_name="unique_key") == ("txn_id", "user_id")
    with pytest.raises(ValueError, match="duplicate"):
        _normalize_key(("a", "a"), field_name="unique_key")
    with pytest.raises(ValueError, match="non-empty"):
        _normalize_key((), field_name="unique_key")
    with pytest.raises(ValueError, match="non-empty"):
        _normalize_key(("a", "  "), field_name="unique_key")
    with pytest.raises(ValueError, match="split_group_key"):
        _normalize_key(123, field_name="split_group_key")
