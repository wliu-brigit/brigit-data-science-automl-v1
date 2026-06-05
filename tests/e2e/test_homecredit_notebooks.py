from __future__ import annotations

import json
import linecache
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.e2e._gates import require_notebook_e2e_env

from automl.mlflow import client as mlflow_client
from automl.project import clear_session

pytestmark = [pytest.mark.e2e, pytest.mark.qa]

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = REPO_ROOT / "projects" / "example_homecredit" / "notebooks"
NOTEBOOKS = [
    NOTEBOOK_DIR / "0_understand_project_sessions_and_routes.ipynb",
    NOTEBOOK_DIR / "1_define_and_materialize_dataset.ipynb",
    NOTEBOOK_DIR / "2_profile_logged_dataset.ipynb",
    NOTEBOOK_DIR / "3.1_run_agent_automl.ipynb",
    NOTEBOOK_DIR / "3.2_author_new_trial.ipynb",
    NOTEBOOK_DIR / "3.3_fork_existing_trial.ipynb",
    NOTEBOOK_DIR / "4_reevaluate_existing_model.ipynb",
    NOTEBOOK_DIR / "5_inspect_logged_runs_and_artifacts.ipynb",
]

def _code_cells(path: Path) -> list[tuple[int, str]]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [
        (index, "".join(cell.get("source", [])))
        for index, cell in enumerate(notebook.get("cells", []), start=1)
        if cell.get("cell_type") == "code"
    ]


def _execute_notebook(path: Path) -> None:
    namespace = {
        "__name__": f"__notebook_e2e_{path.stem}__",
        "__file__": str(path),
    }
    cached_filenames = []
    try:
        for index, source in _code_cells(path):
            stripped = source.lstrip()
            if stripped.startswith(("%", "!")):
                pytest.fail(f"{path.name} cell {index} uses notebook-only shell/magic syntax")
            filename = f"{path}#cell-{index}"
            linecache.cache[filename] = (
                len(source),
                None,
                source.splitlines(keepends=True),
                filename,
            )
            cached_filenames.append(filename)
            exec(compile(source, filename, "exec"), namespace)
    finally:
        for filename in cached_filenames:
            linecache.cache.pop(filename, None)


def test_homecredit_notebooks_execute_end_to_end(monkeypatch):
    require_notebook_e2e_env()
    # Run from the notebook directory, exactly as a Jupyter kernel would —
    # the notebooks rely on cwd-based project inference (the repo now holds
    # several projects, so inferring from the repo root is ambiguous).
    monkeypatch.chdir(NOTEBOOK_DIR)
    namespace = os.environ.get("AUTOML_NOTEBOOK_NAMESPACE") or (
        "qa/notebook-e2e-" + datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
    )
    if not namespace.startswith("qa/"):
        pytest.fail("AUTOML_NOTEBOOK_NAMESPACE for e2e runs must start with 'qa/'")
    monkeypatch.setenv("AUTOML_NOTEBOOK_NAMESPACE", namespace)

    try:
        for path in NOTEBOOKS:
            _execute_notebook(path)
    finally:
        clear_session()
        mlflow_client.clear()
