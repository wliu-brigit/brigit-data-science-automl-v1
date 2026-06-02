from __future__ import annotations

import pytest

from automl.agent.checks import proposal_schema

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


def _checks(issues):
    return {issue.check for issue in issues}


def test_proposal_schema_accepts_valid_proposal_and_required_preprocessing(monkeypatch):
    monkeypatch.setattr("automl.agent.checks.allowed_dependencies", lambda session=None: ["pandas"])

    issues = proposal_schema(
        _valid_payload(
            required_preprocessing=[{"name": "WOEEncoder", "columns": ["ORGANIZATION_TYPE"]}]
        )
    )

    assert issues == []


def test_proposal_schema_reports_required_format_and_removed_fields(monkeypatch):
    monkeypatch.setattr("automl.agent.checks.allowed_dependencies", lambda session=None: ["pandas"])

    issues = proposal_schema(
        {
            "schema_version": "2",
            "slug": "BadSlug",
            "strategy": "",
            "hypothesis": "",
            "implementation_plan": [],
            "constraints": {"not": "a-list"},
            "required_dependencies": [],
            "seed_hint": "run:abc",
            "parent_id": "removed_parent",
        }
    )

    assert {
        "proposal.bad_schema_version",
        "proposal.bad_slug",
        "proposal.bad_strategy",
        "proposal.bad_hypothesis",
        "proposal.bad_implementation_plan",
        "proposal.bad_constraints",
        "proposal.bad_required_dependencies",
        "proposal.bad_seed_hint",
        "proposal.parent_id_removed",
    } <= _checks(issues)


def test_proposal_schema_reports_missing_required_fields(monkeypatch):
    monkeypatch.setattr("automl.agent.checks.allowed_dependencies", lambda session=None: ["pandas"])

    issues = proposal_schema({"schema_version": 2})

    assert "proposal.missing_field" in _checks(issues)
    assert {issue.location for issue in issues if issue.check == "proposal.missing_field"} == {
        "slug",
        "strategy",
        "hypothesis",
        "implementation_plan",
        "constraints",
        "required_dependencies",
    }


def test_proposal_schema_warns_for_unknown_fields_and_checks_dependencies(monkeypatch):
    monkeypatch.setattr("automl.agent.checks.allowed_dependencies", lambda session=None: ["pandas"])

    issues = proposal_schema(
        _valid_payload(required_dependencies=["pandas", "lightgbm"], future_field=True)
    )

    assert ("warning", "proposal.unknown_field", "future_field") in [
        (issue.level, issue.check, issue.location) for issue in issues
    ]
    assert ("error", "proposal.dep_not_allowed", "required_dependencies") in [
        (issue.level, issue.check, issue.location) for issue in issues
    ]


def test_proposal_schema_rejects_non_object_before_field_checks(monkeypatch):
    monkeypatch.setattr("automl.agent.checks.allowed_dependencies", lambda session=None: ["pandas"])

    issues = proposal_schema(["not", "object"])

    assert [(issue.level, issue.check, issue.message) for issue in issues] == [
        ("error", "proposal.not_object", "proposal must be a JSON object")
    ]
