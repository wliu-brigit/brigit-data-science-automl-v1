import importlib

import pytest

pytestmark = pytest.mark.unit

CLI_MODULES = ["project", "experiment", "trial", "data", "eval", "validate", "_common"]
INCIDENTAL = {"Path", "subprocess", "use_project", "argparse"}


def test_cli_all_has_no_incidental_imports():
    for name in CLI_MODULES:
        mod = importlib.import_module(f"automl.cli.{name}")
        leaked = INCIDENTAL.intersection(getattr(mod, "__all__", []))
        assert not leaked, f"automl.cli.{name}.__all__ leaks incidental imports: {leaked}"
