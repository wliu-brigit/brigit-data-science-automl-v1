from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def test_readme_python_requirement_matches_pyproject():
    pyproject = tomllib.loads(_read("pyproject.toml"))
    required = pyproject["project"]["requires-python"]
    readme = _read("README.md")

    assert f"Python {required}" in readme
    assert "Python >=3.13" not in readme
    assert "Python ≥3.13" not in readme


def test_readme_library_guidance_imports_domain_modules_explicitly():
    readme = _read("README.md")

    assert "from automl import data, experiment, trial, eval" in readme
    assert "automl.data.materialize" not in readme
    assert "automl.experiment.leaderboard" not in readme
    assert "automl.trial.show_trial" not in readme
    assert "automl.eval.evaluate" not in readme


def test_active_guidance_uses_proposal_noun_not_trialproposal():
    roots = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "agent-skills" / "skills",
        REPO_ROOT / "agent-skills" / "agents",
        REPO_ROOT / "agent-skills" / "references",
    ]
    offenders = []
    for root in roots:
        paths = [root] if root.is_file() else sorted(root.rglob("*.md"))
        for path in paths:
            text = path.read_text(encoding="utf-8")
            if re.search(r"\bTrialProposal\b", text):
                offenders.append(path.relative_to(REPO_ROOT).as_posix())

    assert offenders == []


def test_guide_paths_point_at_real_repo_files():
    guide = _read("agent-skills/skills/automl-guide/SKILL.md")

    assert "automl_dev/CLAUDE.md" not in guide
    assert "`CLAUDE.md`" in guide
    assert "`automl/data/sources/`" in guide


def test_project_claude_does_not_depend_on_parent_workspace_guide():
    guide = _read("CLAUDE.md")

    assert "../CLAUDE.md" not in guide
    assert "This is the package-level guide." in guide


def test_setup_docs_describe_project_owned_model_and_eval_surfaces():
    model_contract = _read("agent-skills/references/setup/model-contract.md")
    eval_metric = _read("agent-skills/references/setup/evaluation-metric.md")
    project_config = _read("projects/example_homecredit/config.py")
    project_model = _read("projects/example_homecredit/model/__init__.py")

    assert "Projects that provide a default project-baseline model" in model_contract
    assert "projects/<project_name>/model/__init__.py" in model_contract
    assert "otherwise `config.py` owns `EVAL`" in eval_metric
    assert "ProjectConfig is loaded into Session.config" in project_config
    assert "imports projects.example_homecredit.model.MODEL_CLASS" in project_model
