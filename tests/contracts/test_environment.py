import importlib.util

import pytest

pytestmark = pytest.mark.contract


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def test_runtime_dependencies_are_available():
    modules = [
        "cloudpickle",
        "google.cloud.storage",
        "mlflow",
        "numpy",
        "pandas",
        "pyarrow",
        "sklearn",
    ]

    missing = [name for name in modules if not _module_available(name)]

    assert missing == []
