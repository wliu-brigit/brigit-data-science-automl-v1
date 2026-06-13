"""The discovery output contract — every method emits findings in this shape."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

CONTRACT_COLUMNS = ["user_id", "method", "method_version", "score", "evidence"]


@dataclass(frozen=True)
class Finding:
    """One discovered-suspect user, with the evidence that surfaced it."""
    user_id: str
    score: float = 1.0          # method-local rank/strength; not comparable across methods
    evidence: dict = field(default_factory=dict)


@dataclass(frozen=True)
class FindingSet:
    """All findings from one method run, tagged with the method's identity+version."""
    method: str                 # e.g. "scenario:ring_device_burst" or "graph:residual_ring_members"
    method_version: str
    findings: list[Finding]

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {
                "user_id": str(f.user_id),
                "method": self.method,
                "method_version": self.method_version,
                "score": float(f.score),
                "evidence": f.evidence,
            }
            for f in self.findings
        ]
        return pd.DataFrame(rows, columns=CONTRACT_COLUMNS)
