from __future__ import annotations

import pytest

from automl.trial.timing_summary import (
    build_runner_timing_summary,
    enrich_agent_timing_summary,
    round_seconds,
)

pytestmark = pytest.mark.unit


def _phase_delta(summary: dict) -> float:
    return abs(float(summary["total_seconds"]) - sum(float(value) for value in summary["phases"].values()))


def test_round_seconds_uses_five_decimal_places():
    assert round_seconds(1.234564) == 1.23456
    assert round_seconds(1.234565) == 1.23457
    assert round_seconds(1.234566) == 1.23457


def test_build_runner_timing_summary_keeps_runner_detail_order():
    summary = build_runner_timing_summary(
        {
            "schema_version": 1,
            "unit": "seconds",
            "total_seconds": 37.76375816692598,
            "phases": {
                "model_import": 0.0006142500787973404,
                "data_load": 3.6381852910853922,
                "fit": 0.0066427080892026424,
            },
        }
    )

    assert summary == {
        "schema_version": 2,
        "unit": "seconds",
        "total_seconds": 37.76376,
        "phases": {"runner": 37.76376},
        "phase_details": {
            "runner": {
                "total_seconds": 37.76376,
                "phases": {
                    "model_import": 0.00061,
                    "data_load": 3.63819,
                    "fit": 0.00664,
                },
            }
        },
    }


def test_enrich_agent_timing_summary_outputs_chronological_sections():
    runner = build_runner_timing_summary(
        {
            "schema_version": 1,
            "unit": "seconds",
            "total_seconds": 10.0,
            "phases": {"fit": 7.0},
        }
    )

    summary = enrich_agent_timing_summary(
        runner,
        setup_steps=[{"name": "data_materialize", "duration_s": 2.0, "start_s": 95.0}],
        proposer={"start_s": 100.0, "end_s": 110.0},
        proposal_handoff_steps=[{"name": "validate_proposal", "duration_s": 1.0, "start_s": 111.0}],
        coder={"start_s": 120.0, "end_s": 145.0},
        runner={"start_s": 130.0, "end_s": 140.0},
        publish_s=3.0,
    )

    assert list(summary["phases"]) == [
        "setup",
        "proposer",
        "proposal_handoff",
        "coder_implementation",
        "runner",
        "coder_report",
        "publish",
    ]
    assert summary["phases"] == {
        "setup": 5.0,
        "proposer": 10.0,
        "proposal_handoff": 10.0,
        "coder_implementation": 10.0,
        "runner": 10.0,
        "coder_report": 5.0,
        "publish": 3.0,
    }
    assert summary["phase_details"]["proposer"] == {"total_seconds": 10.0}
    assert summary["phase_details"]["publish"] == {"total_seconds": 3.0}
    assert summary["phase_details"]["runner"]["phases"] == {"fit": 7.0}
    assert set(summary) == {
        "schema_version",
        "unit",
        "total_seconds",
        "phases",
        "phase_details",
    }


def test_enrich_agent_timing_summary_phase_totals_reconcile_to_wall_clock():
    runner = build_runner_timing_summary(
        {
            "schema_version": 1,
            "unit": "seconds",
            "total_seconds": 10.0,
            "phases": {"fit": 7.0},
        }
    )

    summary = enrich_agent_timing_summary(
        runner,
        setup_steps=[{"name": "data_materialize", "duration_s": 2.0, "start_s": 95.0}],
        proposer={"start_s": 100.0, "end_s": 110.0},
        proposal_handoff_steps=[{"name": "validate_proposal", "duration_s": 1.0, "start_s": 112.0}],
        coder={"start_s": 120.0, "end_s": 145.0},
        runner={"start_s": 130.0, "end_s": 140.0},
        publish_s=3.0,
    )

    assert summary["total_seconds"] == 53.0
    assert _phase_delta(summary) <= max(1.0, summary["total_seconds"] * 0.02)
    assert summary["phases"]["setup"] == 5.0
    assert summary["phase_details"]["setup"]["phases"] == {
        "data_materialize": 2.0,
        "other": 3.0,
    }
    assert summary["phases"]["proposal_handoff"] == 10.0
    assert summary["phase_details"]["proposal_handoff"]["phases"] == {
        "validate_proposal": 1.0,
        "other": 9.0,
    }


def test_enrich_agent_timing_summary_uses_runner_total_when_runner_span_differs():
    runner = build_runner_timing_summary(
        {
            "schema_version": 1,
            "unit": "seconds",
            "total_seconds": 10.0,
            "phases": {"fit": 7.0},
        }
    )

    summary = enrich_agent_timing_summary(
        runner,
        setup_steps=[{"name": "data_materialize", "duration_s": 2.0, "start_s": 95.0}],
        proposer={"start_s": 100.0, "end_s": 110.0},
        proposal_handoff_steps=[{"name": "validate_proposal", "duration_s": 1.0, "start_s": 112.0}],
        coder={"start_s": 120.0, "end_s": 145.0},
        runner={"start_s": 130.0, "end_s": 142.0},
        publish_s=3.0,
    )

    assert summary["total_seconds"] == 53.0
    assert _phase_delta(summary) <= max(1.0, summary["total_seconds"] * 0.02)
    assert summary["phases"]["runner"] == 10.0
    assert summary["phases"]["coder_report"] == 5.0


def test_enrich_agent_timing_summary_derives_coder_split_without_runner_boundaries():
    runner = build_runner_timing_summary(
        {"schema_version": 1, "unit": "seconds", "total_seconds": 7.0, "phases": {}}
    )

    summary = enrich_agent_timing_summary(
        runner,
        setup_steps=[],
        proposer={"start_s": 10.0, "end_s": 12.0},
        proposal_handoff_steps=[],
        coder={"start_s": 20.0, "end_s": 30.0},
        runner=None,
        publish_s=0.5,
    )

    assert summary["phases"]["coder_implementation"] == 3.0
    assert summary["phases"]["coder_report"] == 0.0
    assert summary["phase_details"]["setup"] == {"total_seconds": 0.0}
    assert summary["phase_details"]["proposal_handoff"] == {
        "total_seconds": 8.0,
        "phases": {"other": 8.0},
    }
