import pytest

import automl.utils as u

pytestmark = pytest.mark.unit


def test_utils_all_names_resolve():
    for name in u.__all__:
        assert hasattr(u, name), f"automl.utils.__all__ lists unresolvable name: {name}"
