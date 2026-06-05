from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]

# The only places the placeholder marker may appear in a scaffolded config.py.
# The placeholder check flags any file containing "TBD_", so a comment that
# spells it would fail `validate project` forever — even after the user fills
# every real slot.
CONFIG_PLACEHOLDERS = (
    "<TBD_target_column>",
    "<TBD_base_table>",
    "<TBD_unique_key>",
    "TBD_experiment_id",
)


def _touch_project(root, name: str) -> None:
    project_dir = root / "projects" / name
    project_dir.mkdir(parents=True)
    (project_dir / "config.py").write_text("# config\n", encoding="utf-8")


def test_list_projects_and_infer_from_cwd_or_single_project(tmp_path, monkeypatch):
    from automl.project.metadata import infer_project_name, list_projects

    _touch_project(tmp_path, "alpha")
    _touch_project(tmp_path, "beta")

    assert list_projects(repo_root=tmp_path) == ["alpha", "beta"]
    assert infer_project_name(repo_root=tmp_path, start=tmp_path / "projects" / "alpha") == "alpha"

    single = tmp_path / "single"
    _touch_project(single, "only")
    monkeypatch.chdir(single)
    assert infer_project_name(repo_root=single) == "only"


def test_infer_project_name_rejects_ambiguous_repo_root(tmp_path, monkeypatch):
    from automl.errors import ProjectError
    from automl.project.metadata import infer_project_name

    _touch_project(tmp_path, "alpha")
    _touch_project(tmp_path, "beta")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ProjectError, match="multiple projects"):
        infer_project_name(repo_root=tmp_path)


def test_create_project_scaffolds_new_import_paths(tmp_path):
    from automl.project.scaffold import create_project

    result = create_project("new_project", project_root=tmp_path)
    config = tmp_path / "projects" / "new_project" / "config.py"

    assert result["project"] == "new_project"
    assert config.exists()
    assert (tmp_path / "projects" / "new_project" / "PROJECT_INSTRUCTIONS.md").exists()
    base_table_sql = tmp_path / "projects" / "new_project" / "data" / "queries" / "base_table.sql"
    assert base_table_sql.exists()
    # The scaffolded SELECT carries its own placeholder (config placeholders
    # stay config-only — see CONFIG_PLACEHOLDERS).
    assert "<TBD_SOURCE_TABLE>" in base_table_sql.read_text(encoding="utf-8")
    text = config.read_text(encoding="utf-8")
    assert "automl.core" not in text
    assert "from automl.project import" in text
    assert "from automl.data import" in text
    assert "from automl.eval import" in text
    assert "PROJECT_CONFIG = ProjectConfig.partial" in text


def test_scaffolded_config_executes_against_library(tmp_path):
    """The scaffold is live code: importing it constructs real library objects.

    This is the drift ratchet for the template — renaming a field or class in
    the library breaks this test until the template is updated to match.
    """
    from automl.data import DataSpec
    from automl.eval import EvalSpec
    from automl.project import BinaryClassification, ProjectConfig, RunConfig
    from automl.project.scaffold import create_project

    create_project("new_project", project_root=tmp_path)
    config_path = tmp_path / "projects" / "new_project" / "config.py"

    spec = importlib.util.spec_from_file_location("scaffolded_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert isinstance(module.TASK, BinaryClassification)
    assert isinstance(module.DATA, DataSpec)
    assert isinstance(module.EVAL, EvalSpec)
    assert isinstance(module.RUN_CONFIG, RunConfig)
    assert isinstance(module.PROJECT_CONFIG, ProjectConfig)


def test_scaffolded_config_placeholders_are_the_only_tbd_occurrences(tmp_path):
    from automl.project.scaffold import create_project

    create_project("new_project", project_root=tmp_path)
    text = (tmp_path / "projects" / "new_project" / "config.py").read_text(encoding="utf-8")

    for placeholder in CONFIG_PLACEHOLDERS:
        assert placeholder in text
        text = text.replace(placeholder, "")
    assert "TBD_" not in text, (
        "TBD_ outside the known placeholders; the placeholder check would flag "
        "the config even after the user fills every slot"
    )


def test_scaffolded_config_comments_stay_truthful(tmp_path):
    """Commented-out alternatives and doc pointers can't be executed, so pin
    them mechanically: every commented constructor's class must be imported by
    the file (so the executable test catches renames), and every reference doc
    the comments point at must exist in the repo."""
    from automl.project.scaffold import create_project

    create_project("new_project", project_root=tmp_path)
    text = (tmp_path / "projects" / "new_project" / "config.py").read_text(encoding="utf-8")

    commented_classes = set(re.findall(r"^# \w+ = (?:EvalSpec\(primary=)?([A-Z]\w+)\(", text, re.M))
    assert commented_classes  # the template lists alternatives; losing them all is a regression
    for cls in commented_classes:
        assert re.search(rf"^(from automl[.\w]* import .*\b{cls}\b|    {cls},)", text, re.M), (
            f"commented alternative {cls} is not imported by the scaffold"
        )

    doc_paths = set(re.findall(r"agent-skills/references/setup/[\w-]+\.md", text))
    assert doc_paths  # the template points at the deeper reference docs
    for doc in doc_paths:
        assert (REPO_ROOT / doc).is_file(), f"scaffold comment points at missing doc: {doc}"


def test_create_project_rejects_invalid_or_existing_project(tmp_path):
    from automl.project.scaffold import create_project

    with pytest.raises(ValueError, match="lower snake_case"):
        create_project("BadName", project_root=tmp_path)

    create_project("demo", project_root=tmp_path)
    with pytest.raises(FileExistsError):
        create_project("demo", project_root=tmp_path)
