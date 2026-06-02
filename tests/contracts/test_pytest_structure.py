import ast
import re
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

REPO_ROOT = Path(__file__).resolve().parents[2]
TIER_MARKERS = {
    "unit": {"unit"},
    "integration": {"integration"},
    "contracts": {"contract"},
    "e2e": {"e2e", "qa"},
}
PHASE_TOKEN = re.compile(
    r"\b(?:phase[-_ ]?[0-9]|AUTOML_PHASE_?[0-9]_E2E)\b",
    re.IGNORECASE,
)
RETIRED_MIGRATION_ENV_TOKEN = re.compile(
    r"\b(?:AUTOML_AUTO_CONFIRM|MLFLOW_BACKEND_STORE_URI|MLFLOW_ARTIFACTS_DESTINATION|"
    r"AUTOML_E2E_(?!NOTEBOOKS\b)[A-Z0-9_]+)\b"
)
ACTIVE_SURFACE_FILES = (
    REPO_ROOT / ".env.example",
    REPO_ROOT / "README.md",
    REPO_ROOT / "CLAUDE.md",
    REPO_ROOT / "docs" / "execution" / "README.md",
)
ACTIVE_SURFACE_DIRS = (
    REPO_ROOT / "automl",
    REPO_ROOT / "tests",
    REPO_ROOT / "projects" / "example_homecredit",
)
ACTIVE_SURFACE_EXCLUDES = {
    REPO_ROOT / "tests" / "contracts" / "test_pytest_structure.py",
}


def test_default_pytest_testpaths_include_all_cutover_tiers():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    testpaths = pyproject["tool"]["pytest"]["ini_options"]["testpaths"]

    assert "tests/unit" in testpaths
    assert "tests/contracts" in testpaths
    assert "tests/integration" in testpaths
    assert "tests/e2e" in testpaths


def _test_files(tier: str) -> list[Path]:
    return sorted((REPO_ROOT / "tests" / tier).rglob("test_*.py"))


def _pytestmark_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "pytestmark" for target in node.targets
        ):
            continue
        values = node.value.elts if isinstance(node.value, (ast.List, ast.Tuple)) else [node.value]
        for value in values:
            if (
                isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Attribute)
                and isinstance(value.value.value, ast.Name)
                and value.value.value.id == "pytest"
                and value.value.attr == "mark"
            ):
                names.add(value.attr)
    return names


def test_test_files_declare_their_tier_marker():
    offenders = []
    for tier, expected in TIER_MARKERS.items():
        for path in _test_files(tier):
            actual = _pytestmark_names(path)
            missing = expected - actual
            if missing:
                offenders.append((path.relative_to(REPO_ROOT).as_posix(), sorted(missing)))

    assert offenders == []


def test_e2e_tests_are_named_by_domain_not_phase():
    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in _test_files("e2e")
        if "phase" in path.name.lower()
    ]

    assert offenders == []


def test_active_surfaces_do_not_keep_temporary_phase_tokens():
    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in _active_surface_files()
        if PHASE_TOKEN.search(path.read_text(encoding="utf-8"))
    ]

    assert offenders == []


def test_active_surfaces_do_not_keep_retired_migration_env_flags():
    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in _active_surface_files()
        if RETIRED_MIGRATION_ENV_TOKEN.search(path.read_text(encoding="utf-8"))
    ]

    assert offenders == []


def test_active_test_guidance_describes_shared_e2e_flags():
    guidance = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")

    assert "AUTOML_E2E=1" in guidance
    assert "AUTOML_E2E_NOTEBOOKS=1" in guidance
    assert "phase environment flags" not in guidance


def test_empty_uncollected_test_tiers_do_not_exist():
    offenders = [
        relative
        for relative in ("tests/shared", "tests/regression")
        if (REPO_ROOT / relative).exists()
    ]

    assert offenders == []


def _active_surface_files() -> list[Path]:
    files = [path for path in ACTIVE_SURFACE_FILES if path.exists()]
    for root in ACTIVE_SURFACE_DIRS:
        if root.is_file():
            files.append(root)
            continue
        files.extend(
            path
            for path in root.rglob("*")
            if path.suffix in {".py", ".md", ".ipynb"}
            and "experiments" not in path.relative_to(root).parts
        )
    return sorted(path for path in files if path not in ACTIVE_SURFACE_EXCLUDES)
