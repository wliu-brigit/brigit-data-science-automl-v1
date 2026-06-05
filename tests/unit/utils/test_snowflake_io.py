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
    parent = types.ModuleType("snowflake")
    parent.connector = fake  # sys.modules cache hits skip the parent-attr wiring
    monkeypatch.setitem(sys.modules, "snowflake", parent)
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


def test_fetch_one_returns_first_row(monkeypatch, fake_connector):
    _set_env(monkeypatch)
    assert sf.fetch_one("SELECT COUNT(*) FROM T") == (1,)
    assert fake_connector["executed"] == ["SELECT COUNT(*) FROM T"]


def test_check_connection_runs_select_1(monkeypatch, fake_connector):
    _set_env(monkeypatch)
    sf.check_connection()
    assert fake_connector["executed"] == ["SELECT 1"]


def test_missing_env_raises_before_any_connection(monkeypatch):
    monkeypatch.delenv("SNOWFLAKE_ACCOUNT", raising=False)
    with pytest.raises(EnvironmentError, match="SNOWFLAKE_ACCOUNT"):
        sf.execute("SELECT 1")


def test_coerce_decimal_columns_casts_computed_number_columns():
    # Snowflake returns computed expressions (HASH, aggregates) as Decimal
    # objects; stored columns arrive as real dtypes. Coercion happens once at
    # the fetch seam so a Decimal can never reach MLflow's JSON encoder.
    from decimal import Decimal

    import pandas as pd

    from automl.utils.io.snowflake import coerce_decimal_columns

    df = pd.DataFrame(
        {
            "computed_int": [Decimal("42"), Decimal("7"), None],
            "computed_frac": [Decimal("1.5"), Decimal("2.25"), Decimal("0")],
            "stored_int": [1, 2, 3],
            "text": ["a", "b", None],
        }
    )
    out = coerce_decimal_columns(df)
    assert pd.api.types.is_numeric_dtype(out["computed_int"])
    assert pd.api.types.is_numeric_dtype(out["computed_frac"])
    assert out["computed_int"].tolist()[:2] == [42, 7]
    assert out["computed_frac"].tolist() == [1.5, 2.25, 0.0]
    assert out["stored_int"].dtype == df["stored_int"].dtype  # untouched
    assert out["text"].dtype == object  # non-Decimal object columns untouched
