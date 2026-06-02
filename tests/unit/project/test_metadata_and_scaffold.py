from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


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
    assert (tmp_path / "projects" / "new_project" / "data" / "queries" / "base_data.sql").exists()
    text = config.read_text(encoding="utf-8")
    assert "automl.core" not in text
    assert "from automl.project import" in text
    assert "from automl.data import" in text
    assert "from automl.eval import" in text
    assert "PROJECT_CONFIG = ProjectConfig.partial" in text


def test_create_project_rejects_invalid_or_existing_project(tmp_path):
    from automl.project.scaffold import create_project

    with pytest.raises(ValueError, match="lower snake_case"):
        create_project("BadName", project_root=tmp_path)

    create_project("demo", project_root=tmp_path)
    with pytest.raises(FileExistsError):
        create_project("demo", project_root=tmp_path)
