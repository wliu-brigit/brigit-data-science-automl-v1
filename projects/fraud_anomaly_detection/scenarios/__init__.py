"""Bound scenario register — static glue, never edited per-scenario.

The definitions live in register.yaml (the file to edit); the machinery
lives in engine.py. This module just binds the two so consumers
(the fit gate, the eval metrics, the backtests) have one stable import:

    from projects.fraud_anomaly_detection.scenarios import assign, residual_mask
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from projects.fraud_anomaly_detection.scenarios import engine
from projects.fraud_anomaly_detection.scenarios.engine import Scenario

REGISTER_PATH = Path(__file__).resolve().parent / "register.yaml"

_REGISTER = engine.load_register(REGISTER_PATH)

SCENARIOS: tuple[Scenario, ...] = _REGISTER.scenarios
SCENARIOS_VERSION: str = _REGISTER.version
TRIGGER_COLUMNS: tuple[str, ...] = _REGISTER.trigger_columns  # derived from the conditions


def assign(df: pd.DataFrame) -> pd.DataFrame:
    """Flag frame for the registered scenarios: scenario_<name> + scenario_any."""
    return engine.evaluate(df, SCENARIOS)


def residual_mask(df: pd.DataFrame) -> pd.Series:
    """True for rows no registered scenario matched — the model's population."""
    return engine.residual(df, SCENARIOS)


__all__ = [
    "REGISTER_PATH",
    "SCENARIOS",
    "SCENARIOS_VERSION",
    "TRIGGER_COLUMNS",
    "Scenario",
    "assign",
    "residual_mask",
]
