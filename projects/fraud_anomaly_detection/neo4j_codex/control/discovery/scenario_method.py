"""Scenario-register adapter for discovery methods."""
from __future__ import annotations

from pathlib import Path

import duckdb

from projects.fraud_anomaly_detection.neo4j_codex.control.contract import Finding, FindingSet
from projects.fraud_anomaly_detection.neo4j_codex.control.discovery.metadata import (
    MethodMetadata,
)
from projects.fraud_anomaly_detection.scenarios import SCENARIOS_VERSION, assign


class ScenarioMethod:
    def __init__(self, scenario_name: str):
        self.scenario_name = scenario_name
        self.metadata = MethodMetadata(
            name=f"scenario:{scenario_name}",
            version=SCENARIOS_VERSION,
            method_type="scenario",
            time_semantics="production_safe",
            promotion_tier="plug_candidate",
            enforcement_projection="scenario_rule",
            params={"scenario_name": scenario_name},
        )
        self.name = self.metadata.name

    def run(self, store: Path | str) -> FindingSet:
        with duckdb.connect(str(store), read_only=True) as con:
            advances = con.execute("SELECT * FROM advances").df()

        flags = assign(advances)
        scenario_column = f"scenario_{self.scenario_name}"
        hit_users = (
            flags.loc[flags[scenario_column].fillna(False), "user_id"]
            if "user_id" in flags.columns
            else advances.loc[flags[scenario_column].fillna(False), "user_id"]
        )
        findings = [
            Finding(user_id=str(user_id), evidence={"scenario": self.scenario_name})
            for user_id in hit_users.astype(str).unique().tolist()
        ]
        return FindingSet(method=self.name, method_version=SCENARIOS_VERSION, findings=findings)
