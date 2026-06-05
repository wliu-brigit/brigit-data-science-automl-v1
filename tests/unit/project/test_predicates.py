"""Where builder + serializable predicate AST."""

import pandas as pd
import pytest

from automl.project.predicates import Predicate, Where

pytestmark = pytest.mark.unit


def _frame():
    return pd.DataFrame(
        {
            "SPLIT_PCT": [5, 50, 95, 20],
            "application_date": ["2026-01-01", "2026-04-01", "2026-02-15", None],
            "amount": [10.0, 20.0, 30.0, 40.0],
        }
    )


def test_comparison_ops_build_leaf_nodes():
    predicate = Where("SPLIT_PCT") < 80
    assert predicate.to_dict() == {"op": "<", "column": "SPLIT_PCT", "value": 80}


@pytest.mark.parametrize(
    "predicate, expected_rows",
    [
        (Where("SPLIT_PCT") < 80, [0, 1, 3]),
        (Where("SPLIT_PCT") >= 80, [2]),
        (Where("SPLIT_PCT") == 50, [1]),
        (Where("SPLIT_PCT") != 50, [0, 2, 3]),
        (Where("application_date") < "2026-03-01", [0, 2]),
        (Where("amount").isin([10.0, 40.0]), [0, 3]),
        (Where("amount").notin([10.0, 40.0]), [1, 2]),
        (Where("application_date").is_null(), [3]),
        (Where("application_date").not_null(), [0, 1, 2]),
        ((Where("SPLIT_PCT") < 80) & (Where("amount") > 15.0), [1, 3]),
        ((Where("SPLIT_PCT") >= 80) | (Where("amount") < 15.0), [0, 2]),
        (~(Where("SPLIT_PCT") < 80), [2]),
    ],
)
def test_mask_selects_expected_rows(predicate, expected_rows):
    df = _frame()
    assert list(df.index[predicate.mask(df)]) == expected_rows


def test_round_trip_through_the_record_form():
    predicate = (Where("application_date") >= "2026-03-01") & (Where("SPLIT_PCT") < 50)
    rebuilt = Predicate.from_dict(predicate.to_dict())
    assert rebuilt == predicate
    assert rebuilt.to_dict() == predicate.to_dict()


def test_missing_column_fails_loudly_at_evaluation():
    with pytest.raises(KeyError, match="no_such_column"):
        (Where("no_such_column") < 1).mask(_frame())


def test_columns_lists_every_referenced_column():
    predicate = (Where("a") < 1) & ((Where("b") == 2) | ~Where("c").is_null())
    assert predicate.columns() == frozenset({"a", "b", "c"})


def test_to_pyarrow_filters_a_table_identically():
    pyarrow = pytest.importorskip("pyarrow")
    df = _frame()
    predicate = (Where("SPLIT_PCT") < 80) & (Where("amount") > 15.0)
    table = pyarrow.Table.from_pandas(df)
    filtered = table.filter(predicate.to_pyarrow()).to_pandas()
    assert sorted(filtered["amount"].tolist()) == sorted(
        df[predicate.mask(df)]["amount"].tolist()
    )


def test_values_must_be_json_scalars():
    with pytest.raises(TypeError, match="JSON"):
        Where("a") < object()


def test_repr_reads_like_the_declaration():
    assert repr(Where("SPLIT_PCT") < 80) == 'Where("SPLIT_PCT") < 80'
    assert "&" in repr((Where("a") < 1) & (Where("b") == 2))
