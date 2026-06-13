"""All tunable thresholds in one place (PRINCIPLES P6) — never baked into facts."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ControlConfig:
    block_tier_precision: float = 0.8     # tau: min DPD45 precision for a plug
    min_support: int = 3                  # min users touching a key to consider it
    min_coverage: int = 2                 # min discovered-fraud users a key must cover
    min_corroborating_types: int = 2      # multi-key corroboration bar
    holdout_days: int = 30                # the two-state holdout window
