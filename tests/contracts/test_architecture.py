import ast
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

REPO_ROOT = Path(__file__).resolve().parents[2]
AUTOML_ROOT = REPO_ROOT / "automl"

EXPECTED_FILES = [
    "__init__.py",
    "errors.py",
]

EXPECTED_DIRS = [
    "project",
    "data",
    "data/sources",
    "model",
    "eval",
    "runner",
    "experiment",
    "experiment/views",
    "trial",
    "agent",
    "mlflow",
    "mlflow/trial/artifacts",
    "validate",
    "utils",
    "utils/io",
    "cli",
]

ALLOWED_TOP_LEVEL = {
    "__init__.py",
    "ARCHITECTURE.md",
    "agent",
    "cli",
    "data",
    "errors.py",
    "eval",
    "experiment",
    "mlflow",
    "model",
    "project",
    "runner",
    "trial",
    "utils",
    "validate",
}

SCAN_ROOTS = ["automl", "projects", "tests"]
IGNORED_PARTS = {"automl_legacy", "docs", ".venv", "__pycache__"}
MLFLOW_TEST_ROOTS = [
    REPO_ROOT / "tests" / "unit" / "mlflow",
    REPO_ROOT / "tests" / "integration" / "mlflow",
    REPO_ROOT / "tests" / "contracts",
]

AUTOML_DOMAINS = {
    "agent",
    "data",
    "eval",
    "experiment",
    "model",
    "project",
    "runner",
    "trial",
}

KNOWN_PRIVATE_MLFLOW_ROUTING_OFFENDERS = set()
STORAGE_ERROR_RAISE_ALLOWED = {
    "automl/utils/io/gcs.py",
    "automl/data/registry.py",  # data read seam: wraps download failures as StorageError
}
DIRECT_MLFLOW_BIND_ALLOWED = {
    "automl/project/session.py",
}
KNOWN_DOMAIN_RAW_MLFLOW_USAGE = set()


def _python_files_under(*roots: str) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        path = REPO_ROOT / root
        if not path.exists():
            continue
        for file_path in path.rglob("*.py"):
            relative_parts = file_path.relative_to(REPO_ROOT).parts
            if any(part in IGNORED_PARTS for part in relative_parts):
                continue
            files.append(file_path)
    return sorted(files)


def _package_for(file_path: Path) -> tuple[str, ...]:
    relative = file_path.relative_to(REPO_ROOT).with_suffix("")
    parts = relative.parts
    if parts[-1] == "__init__":
        return parts[:-1]
    return parts[:-1]


def _resolve_import_from(file_path: Path, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module

    package_parts = _package_for(file_path)
    keep = len(package_parts) - (node.level - 1)
    if keep < 0:
        return None

    resolved_parts = list(package_parts[:keep])
    if node.module:
        resolved_parts.extend(node.module.split("."))
    return ".".join(resolved_parts)


def _imports_in(file_path: Path) -> list[str]:
    tree = ast.parse(file_path.read_text(), filename=str(file_path))
    imports: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = _resolve_import_from(file_path, node)
            if module is None:
                continue
            imports.append(module)
            if module == "automl":
                imports.extend(f"automl.{alias.name}" for alias in node.names)

    return imports


def _imports_private_mlflow_routing(file_path: Path) -> bool:
    tree = ast.parse(file_path.read_text(), filename=str(file_path))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "automl.mlflow._routing" for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            module = _resolve_import_from(file_path, node)
            if module == "automl.mlflow._routing":
                return True
            if module == "automl.mlflow" and any(alias.name == "_routing" for alias in node.names):
                return True

    return False


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _raw_mlflow_call_lines(file_path: Path) -> list[int]:
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    mlflow_client_names: set[str] = set()
    raw_function_names: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "automl.mlflow.client" and alias.asname:
                    mlflow_client_names.add(alias.asname)
        elif isinstance(node, ast.ImportFrom):
            module = _resolve_import_from(file_path, node)
            if module == "automl.mlflow":
                mlflow_client_names.update(
                    alias.asname or alias.name for alias in node.names if alias.name == "client"
                )
            elif module == "automl.mlflow.client":
                raw_function_names.update(
                    alias.asname or alias.name for alias in node.names if alias.name == "raw"
                )

    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in raw_function_names:
            lines.append(node.lineno)
            continue
        if isinstance(func, ast.Attribute) and func.attr == "raw":
            target = _dotted_name(func.value)
            if target in mlflow_client_names or target == "automl.mlflow.client":
                lines.append(node.lineno)
    return lines


def _automl_domain(imported: str) -> str | None:
    parts = imported.split(".")
    if parts[:1] != ["automl"] or len(parts) < 2:
        return None
    return parts[1]


def _relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def test_cli_verb_files_stay_thin():
    budgets = {
        "automl/cli/project.py": 80,
        "automl/cli/experiment.py": 80,
        "automl/cli/trial.py": 80,
        "automl/cli/data.py": 110,  # +30 for data cache list/prune/clear verbs
        "automl/cli/eval.py": 80,
        "automl/cli/validate.py": 80,
    }
    offenders = []

    for relative_path, max_lines in budgets.items():
        line_count = len((REPO_ROOT / relative_path).read_text().splitlines())
        if line_count > max_lines:
            offenders.append(
                {
                    "path": relative_path,
                    "lines": line_count,
                    "max_lines": max_lines,
                }
            )

    assert offenders == []


def test_fresh_package_shape_exists():
    missing = []
    wrong_type = []

    for relative_path in EXPECTED_FILES:
        path = AUTOML_ROOT / relative_path
        if not path.exists():
            missing.append(relative_path)
        elif not path.is_file():
            wrong_type.append(relative_path)

    for relative_path in EXPECTED_DIRS:
        path = AUTOML_ROOT / relative_path
        if not path.exists():
            missing.append(relative_path)
        elif not path.is_dir():
            wrong_type.append(relative_path)

    assert missing == []
    assert wrong_type == []


def test_legacy_shaped_package_paths_do_not_reappear():
    offenders = []
    for path in AUTOML_ROOT.iterdir():
        if path.name == "__pycache__":
            continue
        if path.name not in ALLOWED_TOP_LEVEL:
            offenders.append(path.name)

    for tests_dir in AUTOML_ROOT.rglob("tests"):
        offenders.append(tests_dir.relative_to(AUTOML_ROOT).as_posix())

    assert offenders == []


def test_trial_artifacts_remains_a_folder_not_a_flat_module():
    artifacts_dir = AUTOML_ROOT / "mlflow" / "trial" / "artifacts"
    flat_module = AUTOML_ROOT / "mlflow" / "trial" / "artifacts.py"

    assert artifacts_dir.is_dir()
    assert not flat_module.exists()


def test_cutover_legacy_trees_are_removed():
    assert not (REPO_ROOT / "automl_legacy").exists()
    assert not (REPO_ROOT / "tests_legacy").exists()


def test_new_code_does_not_import_automl_legacy():
    offenders = []
    for file_path in _python_files_under(*SCAN_ROOTS):
        for imported in _imports_in(file_path):
            if imported == "automl_legacy" or imported.startswith("automl_legacy."):
                offenders.append((_relative(file_path), imported))

    assert offenders == []


def test_mlflow_package_is_imported_only_inside_automl_mlflow_seam():
    offenders = []
    for file_path in _python_files_under(*SCAN_ROOTS):
        if file_path.is_relative_to(AUTOML_ROOT / "mlflow"):
            continue
        if any(file_path.is_relative_to(root) for root in MLFLOW_TEST_ROOTS if root.exists()):
            continue

        for imported in _imports_in(file_path):
            if imported == "mlflow" or imported.startswith("mlflow."):
                offenders.append((_relative(file_path), imported))

    assert offenders == []


def test_layer_dependency_contracts_are_present():
    required = {
        "test_leaf_utilities_do_not_import_automl_domains",
        "test_project_domain_does_not_import_downstream_runtime_domains",
        "test_trial_domain_does_not_import_runner_domain",
        "test_validate_is_a_leaf",
        "test_runner_imports_only_approved_pure_trial_leaves",
        "test_domains_do_not_import_private_mlflow_routing",
    }
    current = {
        name for name, value in globals().items() if name.startswith("test_") and callable(value)
    }

    assert required.issubset(current)


def test_leaf_utilities_do_not_import_automl_domains():
    leaf_files = sorted((AUTOML_ROOT / "utils").rglob("*.py")) + [AUTOML_ROOT / "errors.py"]
    offenders = []

    for file_path in leaf_files:
        for imported in _imports_in(file_path):
            imported_domain = _automl_domain(imported)
            if imported_domain in AUTOML_DOMAINS:
                offenders.append((_relative(file_path), imported))

    assert offenders == []


def test_project_domain_does_not_import_downstream_runtime_domains():
    downstream_domains = {"data", "model", "eval", "runner"}
    offenders = []

    for file_path in sorted((AUTOML_ROOT / "project").rglob("*.py")):
        for imported in _imports_in(file_path):
            imported_domain = _automl_domain(imported)
            if imported_domain in downstream_domains:
                offenders.append((_relative(file_path), imported))

    assert offenders == []


def test_trial_domain_does_not_import_runner_domain():
    offenders = []

    for file_path in sorted((AUTOML_ROOT / "trial").rglob("*.py")):
        for imported in _imports_in(file_path):
            if imported == "automl.runner" or imported.startswith("automl.runner."):
                offenders.append((_relative(file_path), imported))

    assert offenders == []


def test_validate_is_a_leaf():
    # The validate package holds only the vocabulary (Issue, ValidationReport,
    # run_check); validation recipes live with their domains. It must not
    # import anything else from the library.
    offenders = []
    for file_path in sorted((AUTOML_ROOT / "validate").rglob("*.py")):
        for imported in _imports_in(file_path):
            if imported.startswith("automl") and not imported.startswith("automl.validate"):
                offenders.append((_relative(file_path), imported))

    assert offenders == []


def test_runner_imports_only_approved_pure_trial_leaves():
    allowed = {
        "automl.trial.manifest",
        "automl.trial.metadata",
        "automl.trial.paths",
        "automl.trial.timing_summary",
        "automl.trial.types",
    }
    offenders = []

    for file_path in sorted((AUTOML_ROOT / "runner").rglob("*.py")):
        for imported in _imports_in(file_path):
            if imported == "automl.trial" or imported.startswith("automl.trial."):
                if imported not in allowed:
                    offenders.append((_relative(file_path), imported))

    assert offenders == []


def test_runner_import_does_not_eager_load_trial_workflows():
    script = """
import sys
import automl.runner

forbidden = {
    "automl.trial.cleanup",
    "automl.trial.create",
    "automl.trial.fork",
    "automl.trial.packaging",
    "automl.trial.show",
    "automl.trial.template",
}
loaded = sorted(forbidden.intersection(sys.modules))
if loaded:
    raise SystemExit("\\n".join(loaded))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_runner_path_and_template_shims_do_not_exist():
    assert not (AUTOML_ROOT / "runner" / "paths.py").exists()
    assert not (AUTOML_ROOT / "runner" / "template.py").exists()


def test_runner_artifacts_module_stays_thin_facade():
    line_count = len((AUTOML_ROOT / "runner" / "artifacts.py").read_text().splitlines())

    assert line_count < 250


def test_trial_status_has_single_public_owner():
    import automl.runner
    import automl.trial

    assert hasattr(automl.trial, "TrialStatus"), "canonical TrialStatus must live in automl.trial"
    assert not hasattr(automl.runner, "TrialStatus"), (
        "runner must not export a second TrialStatus; TrialResult.status is a plain str"
    )


def test_domains_do_not_import_private_mlflow_routing():
    offenders = []
    for file_path in sorted(AUTOML_ROOT.rglob("*.py")):
        if file_path.is_relative_to(AUTOML_ROOT / "mlflow"):
            continue
        if _imports_private_mlflow_routing(file_path):
            offenders.append(_relative(file_path))

    unexpected = sorted(set(offenders) - KNOWN_PRIVATE_MLFLOW_ROUTING_OFFENDERS)

    assert unexpected == []


def test_storage_error_raises_stay_inside_storage_seams():
    offenders = []
    for file_path in sorted(AUTOML_ROOT.rglob("*.py")):
        relative = _relative(file_path)
        if file_path.is_relative_to(AUTOML_ROOT / "mlflow"):
            continue
        if relative in STORAGE_ERROR_RAISE_ALLOWED:
            continue
        for line in file_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("class StorageError("):
                continue
            if "StorageError(" in line:
                offenders.append(relative)
                break

    assert offenders == []


def test_session_binding_goes_through_mlflow_bound_for():
    offenders = []
    for file_path in sorted(AUTOML_ROOT.rglob("*.py")):
        relative = _relative(file_path)
        if file_path.is_relative_to(AUTOML_ROOT / "mlflow"):
            continue
        if relative in DIRECT_MLFLOW_BIND_ALLOWED:
            continue
        text = file_path.read_text(encoding="utf-8")
        if "mlflow_client.bind(" in text:
            offenders.append(relative)

    assert offenders == []


def test_domain_raw_mlflow_usage_stays_at_known_cleanup_seam():
    offenders = []
    for file_path in sorted(AUTOML_ROOT.rglob("*.py")):
        if file_path.is_relative_to(AUTOML_ROOT / "mlflow"):
            continue
        if _raw_mlflow_call_lines(file_path):
            offenders.append(_relative(file_path))

    unexpected = sorted(set(offenders) - KNOWN_DOMAIN_RAW_MLFLOW_USAGE)

    assert unexpected == []
