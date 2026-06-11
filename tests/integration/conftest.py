"""Integration-tier fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_dataset_cache(tmp_path, monkeypatch):
    """Keep the dataset cache out of the developer's real ~/.cache.

    Integration tests exercise the real read seam (and thus the real
    content-addressed cache); without this, every run leaves never-reused
    fixture entries in the home-directory cache that `automl data cache list`
    then shows to users.
    """
    monkeypatch.setenv("AUTOML_CACHE_DIR", str(tmp_path / "automl-cache"))
