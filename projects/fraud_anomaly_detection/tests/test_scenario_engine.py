"""Scenario engine: YAML loading, condition compilation, mask algebra.

The engine is register-agnostic: it compiles condition specs into pandas
masks and never knows which scenarios exist. Triggers are conjunctive (ALL
conditions hold — the SCENARIOS.md rubric); disqualifiers release a row if
ANY of them holds.
"""

import textwrap

import pandas as pd
import pytest

from projects.fraud_anomaly_detection.scenarios.engine import (
    evaluate,
    load_register,
    residual,
)

pytestmark = pytest.mark.unit


def write_register(tmp_path, body):
    path = tmp_path / "scenarios.yaml"
    path.write_text(textwrap.dedent(body))
    return path


TOY_REGISTER = """\
version: "test.1"
scenarios:
  - name: toy
    title: Toy scenario
    typology: test
    tier: review
    status: draft
    entry_date: "2026-06-06"
    theory: test-only
    trigger:
      - column: x
        op: ">"
        value: 1
"""


def test_load_register_compiles_scenarios(tmp_path):
    register = load_register(write_register(tmp_path, TOY_REGISTER))
    assert register.version == "test.1"
    [toy] = register.scenarios
    assert (toy.name, toy.tier, toy.status, toy.entry_date) == ("toy", "review", "draft", "2026-06-06")
    df = pd.DataFrame({"x": [0, 2]})
    flags = evaluate(df, register.scenarios)
    assert flags["scenario_toy"].tolist() == [False, True]
    assert flags["scenario_any"].tolist() == [False, True]
    assert residual(df, register.scenarios).tolist() == [True, False]


def test_trigger_is_a_conjunction_and_collects_columns(tmp_path):
    register = load_register(write_register(tmp_path, """\
        version: "test.1"
        scenarios:
          - name: conj
            title: t
            typology: t
            tier: review
            status: draft
            entry_date: "2026-06-06"
            theory: t
            trigger:
              - column: x
                op: ">="
                value: 1
              - column: y
                op: "=="
                value: "a"
        """))
    df = pd.DataFrame({"x": [1, 1, 0], "y": ["a", "b", "a"]})
    assert evaluate(df, register.scenarios)["scenario_conj"].tolist() == [True, False, False]
    assert register.trigger_columns == ("x", "y")


def test_hours_between_condition_and_null_safety(tmp_path):
    register = load_register(write_register(tmp_path, """\
        version: "test.1"
        scenarios:
          - name: fresh
            title: t
            typology: t
            tier: review
            status: draft
            entry_date: "2026-06-06"
            theory: t
            trigger:
              - hours_between: [later_ts, earlier_ts]
                op: "<="
                value: 24
        """))
    ts = pd.Timestamp("2026-01-15 12:00:00")
    df = pd.DataFrame(
        {
            "later_ts": [ts, ts, ts],
            # 24h boundary inclusive; NaT never matches; string timestamps coerce
            "earlier_ts": [str(ts - pd.Timedelta(hours=24)), ts - pd.Timedelta(hours=25), pd.NaT],
        }
    )
    assert evaluate(df, register.scenarios)["scenario_fresh"].tolist() == [True, False, False]
    assert register.trigger_columns == ("later_ts", "earlier_ts")


def test_disqualifiers_release_when_any_holds(tmp_path):
    register = load_register(write_register(tmp_path, """\
        version: "test.1"
        scenarios:
          - name: gated
            title: t
            typology: t
            tier: review
            status: draft
            entry_date: "2026-06-06"
            theory: t
            trigger:
              - column: x
                op: ">"
                value: 1
            disqualifiers:
              - column: tenured
                op: "=="
                value: 1
              - column: payroll
                op: "=="
                value: 1
        """))
    df = pd.DataFrame({"x": [2, 2, 2], "tenured": [1, 0, 0], "payroll": [0, 1, 0]})
    # released by either disqualifier independently
    assert evaluate(df, register.scenarios)["scenario_gated"].tolist() == [False, False, True]


def test_null_ops_and_isin(tmp_path):
    register = load_register(write_register(tmp_path, """\
        version: "test.1"
        scenarios:
          - name: nulls
            title: t
            typology: t
            tier: review
            status: draft
            entry_date: "2026-06-06"
            theory: t
            trigger:
              - column: device_id
                op: is_null
              - column: kind
                op: isin
                value: ["a", "b"]
        """))
    df = pd.DataFrame({"device_id": [None, None, "d3"], "kind": ["a", "z", "a"]})
    assert evaluate(df, register.scenarios)["scenario_nulls"].tolist() == [True, False, False]


def test_title_and_typology_are_optional(tmp_path):
    register = load_register(write_register(tmp_path, """\
        version: "test.1"
        scenarios:
          - name: minimal
            tier: review
            status: draft
            entry_date: "2026-06-06"
            theory: test-only
            trigger:
              - column: x
                op: ">"
                value: 1
        """))
    [minimal] = register.scenarios
    assert minimal.title == "minimal"  # defaults to the name
    assert minimal.typology == ""  # filled in at grounding time


def test_bad_specs_fail_loudly(tmp_path):
    with pytest.raises(ValueError, match="unknown op"):
        load_register(write_register(tmp_path, """\
            version: "test.1"
            scenarios:
              - name: bad
                title: t
                typology: t
                tier: review
                status: draft
                entry_date: "2026-06-06"
                theory: t
                trigger:
                  - column: x
                    op: "~="
                    value: 1
            """))
    with pytest.raises(ValueError, match="missing required field"):
        load_register(write_register(tmp_path, """\
            version: "test.1"
            scenarios:
              - name: incomplete
                trigger:
                  - column: x
                    op: ">"
                    value: 1
            """))
    with pytest.raises(ValueError, match="empty trigger"):
        load_register(write_register(tmp_path, """\
            version: "test.1"
            scenarios:
              - name: nocond
                title: t
                typology: t
                tier: review
                status: draft
                entry_date: "2026-06-06"
                theory: t
                trigger: []
            """))
