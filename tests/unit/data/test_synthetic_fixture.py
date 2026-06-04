import pytest

from automl.data.synthetic import make_synthetic_fixture

pytestmark = pytest.mark.unit


def test_make_synthetic_fixture_returns_binary_target_and_registry():
    df, registry = make_synthetic_fixture(rows=12)

    assert len(df) == 12
    assert set(df["target"]).issubset({0, 1})
    assert registry.get("target").target is True
    assert registry.get("value").model is True
