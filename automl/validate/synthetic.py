"""Synthetic validation fixtures."""

from __future__ import annotations

import pandas as pd

from automl.data import FeatureRegistry


def make_synthetic_fixture(rows: int = 50):
    if rows < 2:
        raise ValueError("rows must be at least 2")
    df = pd.DataFrame(
        {
            "target": [index % 2 for index in range(rows)],
            "value": [float(index) for index in range(rows)],
            "noise": [float((index * 7) % 5) for index in range(rows)],
        }
    )
    registry = FeatureRegistry().build_from_df(df, target_column="target")
    return df, registry


__all__ = ["make_synthetic_fixture"]
