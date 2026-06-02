from __future__ import annotations

import os

import pytest


LIVE_E2E_ENV = "AUTOML_E2E"
NOTEBOOK_E2E_ENV = "AUTOML_E2E_NOTEBOOKS"
SERVICE_ENV = ("GCS_BUCKET", "GCP_PROJECT", "MLFLOW_TRACKING_URI")


def _missing(names: tuple[str, ...]) -> list[str]:
    return [name for name in names if not os.environ.get(name)]


def requires_live_e2e(label: str):
    required = (LIVE_E2E_ENV, *SERVICE_ENV)
    return pytest.mark.skipif(
        bool(_missing(required)),
        reason=f"{label} e2e requires {', '.join(required)}",
    )


def require_notebook_e2e_env() -> None:
    required = (NOTEBOOK_E2E_ENV, *SERVICE_ENV)
    missing = _missing(required)
    if missing:
        pytest.skip(f"Home Credit notebook e2e requires {', '.join(required)}")
