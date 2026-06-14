from projects.fraud_anomaly_detection.scenarios import SCENARIOS, SCENARIOS_VERSION
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
