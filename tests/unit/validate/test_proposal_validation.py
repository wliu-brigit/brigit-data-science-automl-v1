from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from automl.validate import Issue, ValidationReport, proposal

pytestmark = pytest.mark.unit


def _valid_payload(**overrides):
    payload = {
        "schema_version": 2,
        "slug": "numeric_baseline",
        "strategy": "baseline",
        "hypothesis": "Establish a numeric baseline.",
        "implementation_plan": ["Fit a simple numeric model."],
        "constraints": ["Do not read test data."],
        "required_dependencies": ["pandas"],
    }
    payload.update(overrides)
    return payload


def test_validation_report_round_trips_schema_and_locations():
    report = ValidationReport(
        issues=[
            Issue(level="warning", check="proposal.unknown_field", message="ignored", location="x")
        ]
    )

    payload = report.to_json()
    payload["future"] = "ignored"

    assert payload["schema_version"] == 1
    assert payload["passed"] is True
    assert ValidationReport.from_dict(payload) == report


def test_validate_proposal_wraps_agent_schema(monkeypatch):
    monkeypatch.setattr("automl.agent.checks.allowed_dependencies", lambda session=None: ["pandas"])

    report = proposal(proposal=_valid_payload())

    assert report.passed
    assert report.to_json()["issues"] == []


def test_validate_proposal_reports_dependency_errors(monkeypatch):
    monkeypatch.setattr("automl.agent.checks.allowed_dependencies", lambda session=None: ["pandas"])

    report = proposal(proposal=_valid_payload(required_dependencies=["lightgbm"]))

    assert not report.passed
    assert report.issues[0].check == "proposal.dep_not_allowed"


def test_allowed_dependencies_reads_project_and_dependency_groups(tmp_path):
    from automl.project.dependencies import allowed_dependencies

    (tmp_path / "pyproject.toml").write_text(
        json.dumps({}),
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
dependencies = ["pandas>=2", "scikit-learn"]

[dependency-groups]
dev = ["pytest", "pandas>=2"]
agent = ["lightgbm>=4"]
""".strip(),
        encoding="utf-8",
    )
    session = SimpleNamespace(config=SimpleNamespace(repo_root=tmp_path))

    assert allowed_dependencies(session) == ["pandas", "scikit-learn", "pytest", "lightgbm"]
