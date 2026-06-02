import logging

import pytest

from automl.utils.logging import configure_logging

pytestmark = pytest.mark.unit


def test_configure_logging_uses_runtime_logger_name(monkeypatch):
    calls = []

    def fake_basic_config(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(logging, "basicConfig", fake_basic_config)

    logger = configure_logging("automl.runner", level=logging.DEBUG)

    assert logger.name == "automl.runner"
    assert calls == [
        {
            "level": logging.DEBUG,
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
        }
    ]
