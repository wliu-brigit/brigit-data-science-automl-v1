from __future__ import annotations

from dataclasses import fields

import pytest

from automl.agent.proposal import DISALLOWED, Proposal, proposal_field_names
from automl.utils.slug import SLUG_RE

pytestmark = pytest.mark.unit


def _valid_payload(**overrides):
    payload = {
        "schema_version": 2,
        "slug": "numeric_baseline",
        "strategy": "baseline",
        "hypothesis": "Establish a numeric baseline.",
        "implementation_plan": ["Fit a simple numeric model."],
        "constraints": ["Do not read test data."],
        "required_dependencies": ["pandas", "scikit-learn"],
        "rationale": "Cold start.",
        "evidence": ["No prior trials."],
        "data_checks": ["Profile has low missingness."],
        "risk_notes": "May underfit.",
        "seed_hint": "auto",
        "required_preprocessing": [{"name": "WOEEncoder", "columns": ["ORGANIZATION_TYPE"]}],
        "ignored_future_field": "ignored",
    }
    payload.update(overrides)
    return payload


def test_proposal_from_dict_strips_unknown_fields_and_round_trips_optional_values():
    proposal = Proposal.from_dict(_valid_payload())

    assert proposal.schema_version == 2
    assert proposal.slug == "numeric_baseline"
    assert proposal.required_preprocessing == [
        {"name": "WOEEncoder", "columns": ["ORGANIZATION_TYPE"]}
    ]
    assert "ignored_future_field" not in proposal.to_dict()
    assert Proposal.from_dict(proposal.to_dict()) == proposal


def test_proposal_roster_is_derived_from_dataclass_fields():
    assert DISALLOWED == ("parent_id",)
    assert proposal_field_names() == tuple(field.name for field in fields(Proposal))
    assert "required_preprocessing" in proposal_field_names()


def test_slug_re_is_shared_lowercase_snake_case_primitive():
    assert SLUG_RE.fullmatch("baseline_v2")
    assert not SLUG_RE.fullmatch("Baseline")
    assert not SLUG_RE.fullmatch("2_baseline")
