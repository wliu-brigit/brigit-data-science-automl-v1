"""Native Neo4j replication of the burst-style scenarios.

Replaces the DuckDB aggregate scenario computation for the scenarios whose logic
is a graph-native burst pattern (>= N distinct users sharing one entity inside a
72h window), derived only from edge `first_ts` + linkage — no upstream feature
column. This collapses the two discovery paths (DuckDB scenarios + Neo4j graph
methods) toward one definition that lives in Neo4j.

Parked: `ring_account_reuse` stays on the DuckDB path for now — its trigger needs
advance-grain fields (loan_amount, is_joint, prior-advance counts) the user-level
mirror does not carry. `NativeScenarioSource` sources parked scenarios from the
DuckDB `truth` frame so the report's scenario set is still complete.

Retrospective semantics: a qualifying 72h window flags ALL its members, including
the first enrollees (we discover after the fact). As-of: when a cutoff is given,
only edges with `first_ts <= cutoff` are considered.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from projects.fraud_anomaly_detection.neo4j_codex.control.discovery.metadata import (
    Source,
    depends_on_label,
    source_label,
)
from projects.fraud_anomaly_detection.neo4j_codex.control.graph.client import (
    GraphQueryRunner,
)
from projects.fraud_anomaly_detection.scenarios import SCENARIOS

# scenario name -> (relationship type, entity node label, min distinct users / 72h)
NATIVE_BURST_SPECS: dict[str, tuple[str, str, int]] = {
    "ring_device_burst": ("USED_DEVICE", "Device", 3),
    "ring_identity_burst": ("USED_BANK_ACCOUNT", "BankAccount", 3),
    "ring_shared_persistent_account": ("USED_PERSISTENT_ACCOUNT", "PersistentAccount", 2),
}

# scenarios kept on the DuckDB path (sourced from the truth frame), not yet native
PARKED_SCENARIOS: frozenset[str] = frozenset({"ring_account_reuse"})

# The feature column(s) each scenario leans on when computed via the DuckDB path
# (declared, for the report's `depends_on`).
DUCKDB_SCENARIO_DEPENDS: dict[str, tuple[str, ...]] = {
    "ring_account_reuse": (
        "column:loan_amount",
        "column:is_joint",
        "column:prior_advances_on_bank_account_7d",
    ),
    "ring_identity_burst": ("column:users_on_bank_account_72h",),
    "ring_shared_persistent_account": (
        "column:users_on_persistent_account_id_72h",
        "column:is_joint",
    ),
    "ring_device_burst": ("column:users_on_device_id_72h",),
}


@dataclass(frozen=True)
class ScenarioDescriptor:
    """Declared source + dependencies for one scenario, for the report's labels."""

    source: Source
    depends_on: tuple[str, ...]

    @property
    def source_label(self) -> str:
        return source_label(self.source)

    @property
    def depends_on_label(self) -> str:
        return depends_on_label(self.depends_on)

_BURST_QUERY = """
MATCH (u:User)-[r:{rel}]->(d:{label})
WHERE ($as_of IS NULL OR r.first_ts <= localdatetime($as_of))
WITH d, collect({{uid: u.user_id, t: r.first_ts}}) AS members
WHERE size(members) >= {thr}
UNWIND members AS a
WITH members, a,
     [x IN members WHERE x.t >= a.t
        AND duration.inSeconds(a.t, x.t).seconds <= 72*3600] AS window
WHERE size(window) >= {thr}
UNWIND window AS w
RETURN collect(DISTINCT w.uid) AS ids
"""


def _as_neo4j_localdatetime(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return pd.Timestamp(value).strftime("%Y-%m-%dT%H:%M:%S.%f")


def burst_users(
    runner: GraphQueryRunner,
    rel_type: str,
    node_label: str,
    threshold: int,
    *,
    as_of: object | None = None,
) -> set[str]:
    """Users in a >= threshold / 72h burst on the given entity type, derived natively."""
    query = _BURST_QUERY.format(rel=rel_type, label=node_label, thr=int(threshold))
    rows = runner.run(query, {"as_of": _as_neo4j_localdatetime(as_of)})
    ids = rows[0]["ids"] if rows and rows[0].get("ids") else []
    return {str(user_id) for user_id in ids}


class NativeScenarioSource:
    """Scenario-set source: native Neo4j for burst scenarios, DuckDB for parked ones.

    Callable with the same shape the report expects: ``(truth, as_of) -> {name: set}``.
    """

    def __init__(self, runner: GraphQueryRunner) -> None:
        self.runner = runner

    def describe(self, scenario_name: str) -> ScenarioDescriptor:
        """Declared source + dependencies for a scenario under the native source."""
        if scenario_name in NATIVE_BURST_SPECS:
            return ScenarioDescriptor(source="neo4j_cypher", depends_on=())  # structural
        return ScenarioDescriptor(
            source="duckdb",
            depends_on=DUCKDB_SCENARIO_DEPENDS.get(scenario_name, ()),
        )

    def __call__(
        self,
        truth: pd.DataFrame,
        *,
        as_of: object | None = None,
    ) -> dict[str, set[str]]:
        sets: dict[str, set[str]] = {}
        for scenario in SCENARIOS:
            spec = NATIVE_BURST_SPECS.get(scenario.name)
            if spec is not None:
                rel, label, threshold = spec
                sets[scenario.name] = burst_users(
                    self.runner, rel, label, threshold, as_of=as_of
                )
            else:
                column = f"scenario_{scenario.name}"
                sets[scenario.name] = (
                    set(truth.index[truth[column]]) if column in truth.columns else set()
                )
        return sets
