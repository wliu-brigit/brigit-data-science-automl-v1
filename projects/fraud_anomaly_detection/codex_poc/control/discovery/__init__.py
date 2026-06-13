"""Discovery method contract for the fraud-control skeleton."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from projects.fraud_anomaly_detection.codex_poc.control.contract import FindingSet


@runtime_checkable
class DiscoveryMethod(Protocol):
    name: str

    def run(self, store: Path | str) -> FindingSet: ...
