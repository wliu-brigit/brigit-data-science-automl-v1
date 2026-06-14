import yaml

from projects.fraud_anomaly_detection.scenarios import SCENARIOS, SCENARIOS_VERSION
from projects.fraud_anomaly_detection.scenarios import REGISTER_PATH
from projects.fraud_anomaly_detection.scenarios.backtest import monthly_backtest


def test_monthly_backtest_scenarios_match_register():
    assert monthly_backtest.REGISTER_VERSION == SCENARIOS_VERSION
    assert [name for name, _ in monthly_backtest.SCENARIOS] == [
        scenario.name for scenario in SCENARIOS
    ]


def test_monthly_backtest_sql_contains_registered_scenario_flags():
    sql = monthly_backtest.build_sql()

    for scenario in SCENARIOS:
        assert f"match_{scenario.name}" in sql
        assert f"'{scenario.name}' AS scenario" in sql


def test_monthly_backtest_predicates_match_register_conditions():
    expected = {
        raw["name"]: _scenario_sql(raw)
        for raw in _raw_register_scenarios()
    }

    assert dict(monthly_backtest.SCENARIOS) == expected


def _raw_register_scenarios() -> list[dict]:
    register_doc = next(yaml.safe_load_all(REGISTER_PATH.read_text()))
    return register_doc["scenarios"]


def _scenario_sql(raw: dict) -> str:
    clauses = [_condition_sql(condition) for condition in raw["trigger"]]
    clauses.extend(
        _negated_disqualifier_sql(condition)
        for condition in raw.get("disqualifiers", [])
    )
    return " AND ".join(clauses)


def _condition_sql(condition: dict) -> str:
    if "hours_between" in condition:
        later, earlier = condition["hours_between"]
        return (
            f"DATEDIFF('second', {earlier}, {later}) / 3600.0 "
            f"{condition['op']} {_sql_value(condition['value'])}"
        )
    return f"{condition['column']} {condition['op']} {_sql_value(condition['value'])}"


def _negated_disqualifier_sql(condition: dict) -> str:
    if condition == {"column": "is_joint", "op": "==", "value": 1}:
        return "COALESCE(is_joint, 0) != 1"
    raise AssertionError(f"unhandled monthly-backtest disqualifier: {condition!r}")


def _sql_value(value: object) -> str:
    if isinstance(value, str):
        return f"'{value}'"
    return str(value)
