from __future__ import annotations

import ast
import json
import re
from dataclasses import fields
from pathlib import Path

import pytest

from automl.eval import EvalResult

pytestmark = pytest.mark.contract

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = REPO_ROOT / "projects" / "example_homecredit" / "notebooks"
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
EXPECTED_NOTEBOOK_NAMES = [
    "0_understand_project_sessions_and_routes.ipynb",
    "1_define_and_materialize_dataset.ipynb",
    "2_profile_logged_dataset.ipynb",
    "3.1_run_agent_automl.ipynb",
    "3.2_author_new_trial.ipynb",
    "3.3_fork_existing_trial.ipynb",
    "4_reevaluate_existing_model.ipynb",
    "5_inspect_logged_runs_and_artifacts.ipynb",
]
EVAL_RESULT_ATTRIBUTES = {field.name for field in fields(EvalResult)} | {"to_dict"}

RETIRED_NOTEBOOK_PATTERNS = {
    "automl.load_project": re.compile(r"\bautoml\.load_project\b"),
    "top-level inspect facade": re.compile(r"from automl import .*inspect|automl\.inspect\b"),
    "top-level profile facade": re.compile(r"from automl import .*profile|automl\.profile\b"),
    "build_pipeline": re.compile(r"\bbuild_pipeline\b|\bdata\.build_pipeline\b"),
    "data source preview method": re.compile(r"\.preview\("),
    "identity preprocessor placeholder": re.compile(r"preprocessor\s*=\s*[\"']identity[\"']"),
    "loaded model run_id attribute": re.compile(r"\bpython_model\.run_id\b"),
    "trial run facade": re.compile(r"\btrial\.run\("),
    "eval publish module": re.compile(r"\bautoml\.eval\.publish\b|from automl\.eval\.publish"),
    "old io facade": re.compile(r"\bautoml\.io\.gcs\b|from automl\.io\.gcs"),
    "domain per-call dry_run": re.compile(
        r"\b(?:data|experiment|trial|eval)\.[A-Za-z_]+\([^)]*\bdry_run\s*=",
        re.DOTALL,
    ),
}


def _notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def _code_cells(path: Path) -> list[str]:
    return [
        _source(cell)
        for cell in _notebook(path).get("cells", [])
        if cell.get("cell_type") == "code"
    ]


def _is_eval_evaluate_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "evaluate"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "eval"
    )


def _target_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        return [
            name
            for item in node.elts
            for name in _target_names(item)
        ]
    return []


class _EvalResultAttributeVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.eval_result_names: set[str] = set()
        self.offenders: list[tuple[int, str, str]] = []

    def visit_Assign(self, node: ast.Assign) -> None:
        if _is_eval_evaluate_call(node.value):
            for target in node.targets:
                self.eval_result_names.update(_target_names(target))
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if _is_eval_evaluate_call(node.value):
            self.eval_result_names.update(_target_names(node.target))
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if (
            isinstance(node.value, ast.Name)
            and node.value.id in self.eval_result_names
            and node.attr not in EVAL_RESULT_ATTRIBUTES
        ):
            self.offenders.append((node.lineno, node.value.id, node.attr))
        self.generic_visit(node)


class _FeatureRegistryBuildVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.offenders: list[int] = []

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "build_from_df"
            and not any(keyword.arg == "target_column" for keyword in node.keywords)
        ):
            self.offenders.append(node.lineno)
        self.generic_visit(node)


def test_homecredit_notebook_first_code_cells_import_clean():
    assert NOTEBOOKS
    for path in NOTEBOOKS:
        first = _code_cells(path)[0]
        namespace = {"__name__": "__notebook_smoke__"}
        exec(compile(first, str(path), "exec"), namespace)


def test_homecredit_notebooks_follow_workflow_order():
    assert [path.name for path in NOTEBOOKS] == EXPECTED_NOTEBOOK_NAMES


def test_homecredit_notebooks_use_final_facade_names():
    offenders = []
    for path in NOTEBOOKS:
        text = path.read_text(encoding="utf-8")
        for label, pattern in RETIRED_NOTEBOOK_PATTERNS.items():
            if pattern.search(text):
                offenders.append((path.relative_to(REPO_ROOT).as_posix(), label))

    assert offenders == []


def test_homecredit_notebooks_use_real_eval_result_attributes():
    offenders = []
    for path in NOTEBOOKS:
        for index, source in enumerate(_code_cells(path), start=1):
            tree = ast.parse(source, filename=f"{path}:cell-{index}")
            visitor = _EvalResultAttributeVisitor()
            visitor.visit(tree)
            for line, name, attr in visitor.offenders:
                offenders.append((path.relative_to(REPO_ROOT).as_posix(), index, line, name, attr))

    assert offenders == []


def test_homecredit_notebooks_pass_required_feature_registry_fields():
    offenders = []
    for path in NOTEBOOKS:
        for index, source in enumerate(_code_cells(path), start=1):
            tree = ast.parse(source, filename=f"{path}:cell-{index}")
            visitor = _FeatureRegistryBuildVisitor()
            visitor.visit(tree)
            for line in visitor.offenders:
                offenders.append((path.relative_to(REPO_ROOT).as_posix(), index, line))

    assert offenders == []


def test_prediction_inspection_follows_recorded_eval_artifact_pointers():
    notebook = (NOTEBOOK_DIR / "5_inspect_logged_runs_and_artifacts.ipynb").read_text(
        encoding="utf-8"
    )

    assert "predictions_prefix" not in notebook
    assert "/predictions/{eval_dataset" not in notebook
    assert "eval_entry.predictions_uri" in notebook
    assert "mlflow_trial.artifacts.load_predictions" in notebook
