"""Fit gate — scenario-matched rows never reach model training.

Trial fit code applies this before fitting (a hard constraint in
PROJECT_INSTRUCTIONS.md): rows a codified scenario already explains are
rule-handled, and the model's job is discovery on the residual. The runner
passes fit the full loaded frame (metadata columns included), so the
scenario predicates can be evaluated here without touching the dataset.

Usage in trial model code::

    from projects.fraud_anomaly_detection.scenarios.gate import gate_fit

    df_train = gate_fit(df_train)
"""

from __future__ import annotations

import pandas as pd

from projects.fraud_anomaly_detection.scenarios import residual_mask


def gate_fit(df: pd.DataFrame) -> pd.DataFrame:
    """Return the residual training frame: rows no scenario matched."""
    return df.loc[residual_mask(df)].reset_index(drop=True)


__all__ = ["gate_fit"]
