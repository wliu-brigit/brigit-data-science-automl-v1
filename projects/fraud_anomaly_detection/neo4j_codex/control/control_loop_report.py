"""Repeatable fraud-control report for scenarios, graph screens, plugs, and holdout."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

from projects.fraud_anomaly_detection.neo4j_codex.control import plug
from projects.fraud_anomaly_detection.neo4j_codex.control.config import ControlConfig
from projects.fraud_anomaly_detection.neo4j_codex.control.discovery.scenario_method import (
    ScenarioMethod,
)
from projects.fraud_anomaly_detection.neo4j_codex.control.discovery.selection import (
    DiscoveryCandidate,
    SelectionRule,
    SelectionRow,
    select_candidates,
)
from projects.fraud_anomaly_detection.neo4j_codex.control.graph.client import Neo4jClient
from projects.fraud_anomaly_detection.neo4j_codex.control.graph.methods import (
    Neo4jGraphDiscovery,
)
from projects.fraud_anomaly_detection.neo4j_codex.control.holdout import two_state_split
from projects.fraud_anomaly_detection.neo4j_codex.control.plug_report import summarize_plugs
from projects.fraud_anomaly_detection.scenarios import SCENARIOS, SCENARIOS_VERSION, assign

DEFAULT_STORE = Path("projects/fraud_anomaly_detection/data/graph/fraud_graph.duckdb")
DEFAULT_OUT_DIR = Path("projects/fraud_anomaly_detection/neo4j_codex/reports")
DEFAULT_REFRESH_KEY = "fraud_control_loop_report"
GRAPH_STATUSES = frozenset(
    {
        "promoted_to_plug_derivation",
        "review_only",
        "below_min_marginal_users",
        "below_min_marginal_dpd45_user_rate",
    }
)


@dataclass(frozen=True)
class ControlLoopReportConfig:
    """Parameters for the report generation and graph-method selection."""

    store: Path = DEFAULT_STORE
    out_dir: Path = DEFAULT_OUT_DIR
    refresh_key: str = DEFAULT_REFRESH_KEY
    graph_min_marginal_users: int = 10
    graph_min_marginal_dpd45_user_rate: float = 0.50
    plug_config: ControlConfig = field(default_factory=ControlConfig)
    include_statuses: frozenset[str] = GRAPH_STATUSES

    def __post_init__(self) -> None:
        unknown = set(self.include_statuses) - GRAPH_STATUSES
        if unknown:
            allowed = ", ".join(sorted(GRAPH_STATUSES))
            raise ValueError(
                f"include_statuses must use {allowed}; got {', '.join(sorted(unknown))}"
            )


@dataclass(frozen=True)
class ReportPaths:
    markdown: Path
    json: Path


def generate_control_loop_report(
    config: ControlLoopReportConfig,
    *,
    graph_discovery: Neo4jGraphDiscovery | None = None,
) -> dict:
    """Generate the control-loop report and write Markdown/JSON files."""
    if not config.store.exists():
        raise FileNotFoundError(f"store not found: {config.store}")
    graph_discovery = graph_discovery or _default_graph_discovery()

    config.out_dir.mkdir(parents=True, exist_ok=True)
    paths = ReportPaths(
        markdown=config.out_dir / f"{config.refresh_key}.md",
        json=config.out_dir / f"{config.refresh_key}.json",
    )

    advances = _load_inputs(config.store)
    truth = _user_truth(advances)
    scenarios = _scenario_sets(truth)
    scenario_candidates = _scenario_candidates(scenarios)
    scenario_union = set().union(*scenarios.values()) if scenarios else set()
    graph_screens = _graph_screen_candidates(graph_discovery, scenarios)
    selected_graphs, excluded_graphs = _select_graph_screens(
        graph_screens,
        scenario_union=scenario_union,
        truth=truth,
        min_marginal_users=config.graph_min_marginal_users,
        min_marginal_dpd45_user_rate=config.graph_min_marginal_dpd45_user_rate,
    )

    selected_graph_union = (
        set().union(*(method.users for method in selected_graphs)) if selected_graphs else set()
    )
    selected_graph_net_new = selected_graph_union - scenario_union
    final_discovery = scenario_union | selected_graph_union

    split = two_state_split(config.store, config.plug_config)
    state_advances = _asof_advances(advances, split.cutoff)
    state_truth = _user_truth(state_advances)
    state_scenarios = _scenario_sets(state_truth)
    state_scenario_union = (
        set().union(*state_scenarios.values()) if state_scenarios else set()
    )
    state_graph_screens = _graph_screen_candidates(
        graph_discovery,
        state_scenarios,
        as_of=split.cutoff,
    )
    selected_state_graphs, _ = _select_graph_screens(
        state_graph_screens,
        scenario_union=state_scenario_union,
        truth=state_truth,
        min_marginal_users=config.graph_min_marginal_users,
        min_marginal_dpd45_user_rate=config.graph_min_marginal_dpd45_user_rate,
    )
    selected_state_graph_union = (
        set().union(*(method.users for method in selected_state_graphs))
        if selected_state_graphs
        else set()
    )
    state_final_discovery = state_scenario_union | selected_state_graph_union
    state_discovery = pd.Series(
        sorted(state_final_discovery & set(split.state_a_users)),
        dtype="string",
    )
    holdout_discovery = pd.Series(
        sorted(final_discovery & set(split.holdout_users)),
        dtype="string",
    )
    stats = plug.candidate_stats(
        config.store,
        state_discovery,
        eligible_users=split.state_a_users,
        end_ts=split.cutoff,
    )
    burned = plug.qualify(stats, config.plug_config)
    state_plug = summarize_plugs(
        config.store,
        burned,
        discovery_users=state_discovery,
        eligible_users=split.state_a_users,
        end_ts=split.cutoff,
    )
    holdout_plug = summarize_plugs(
        config.store,
        burned,
        discovery_users=holdout_discovery,
        eligible_users=split.holdout_users,
        start_ts=split.cutoff,
    )

    scenario_rows = _scenario_rows(scenario_candidates, scenario_union, truth)
    graph_selection_rows = [*selected_graphs, *excluded_graphs]
    graph_rows = [_graph_row(method) for method in graph_selection_rows]
    graph_status_counts = {
        status: sum(1 for row in graph_rows if row["status"] == status)
        for status in sorted(GRAPH_STATUSES)
    }
    included_graph_rows = [
        row for row in graph_rows if row["status"] in config.include_statuses
    ]
    selected_rows = [
        row for row in included_graph_rows if row["status"] == "promoted_to_plug_derivation"
    ]
    excluded_rows = [
        row for row in included_graph_rows if row["status"] != "promoted_to_plug_derivation"
    ]
    review_graph_net_new = (
        set().union(
            *[
                set(method.net_new_users)
                for method in graph_selection_rows
                if _graph_status(method) == "review_only"
            ]
        )
        if graph_selection_rows
        else set()
    )
    state_rows = [
        _plug_bucket_row(bucket, state_plug[bucket])
        for bucket in ["covered_discovery", "uncovered_discovery", "outside_discovery"]
    ]
    holdout_rows = [
        _plug_bucket_row(bucket, holdout_plug[bucket])
        for bucket in ["covered_discovery", "uncovered_discovery", "outside_discovery"]
    ]

    payload = {
        "store": str(config.store),
        "scenario_version": SCENARIOS_VERSION,
        "config": {
            **asdict(config),
            "store": str(config.store),
            "out_dir": str(config.out_dir),
            "plug_config": asdict(config.plug_config),
            "include_statuses": sorted(config.include_statuses),
        },
        "selection_rule": {
            "min_marginal_users": config.graph_min_marginal_users,
            "min_marginal_dpd45_user_rate": config.graph_min_marginal_dpd45_user_rate,
            "include_statuses": sorted(config.include_statuses),
        },
        "scenario_rows": scenario_rows,
        "graph_rows": included_graph_rows,
        "graph_status_counts": graph_status_counts,
        "review_graph_net_new_users": len(review_graph_net_new),
        "review_graph_net_new_dpd45_users": _outcome(review_graph_net_new, truth)[
            "dpd45_users"
        ],
        "review_graph_net_new_dpd45_user_rate": _outcome(review_graph_net_new, truth)[
            "dpd45_user_rate"
        ],
        "review_graph_net_new_dpd45_advances": _outcome(review_graph_net_new, truth)[
            "dpd45_advances"
        ],
        "review_graph_net_new_mature_advances": _outcome(review_graph_net_new, truth)[
            "mature_advances"
        ],
        "review_graph_net_new_dpd45_advance_rate": _outcome(review_graph_net_new, truth)[
            "dpd45_advance_rate"
        ],
        "selected_graph_rows": selected_rows,
        "excluded_graph_rows": excluded_rows,
        "final_discovery": {
            "scenario_union_users": len(scenario_union),
            "selected_graph_net_new_users": len(selected_graph_net_new),
            "selected_graph_net_new_dpd45_users": _outcome(
                selected_graph_net_new, truth
            )["dpd45_users"],
            "selected_graph_net_new_dpd45_user_rate": _outcome(
                selected_graph_net_new, truth
            )["dpd45_user_rate"],
            "selected_graph_net_new_dpd45_advances": _outcome(
                selected_graph_net_new, truth
            )["dpd45_advances"],
            "selected_graph_net_new_mature_advances": _outcome(
                selected_graph_net_new, truth
            )["mature_advances"],
            "selected_graph_net_new_dpd45_advance_rate": _outcome(
                selected_graph_net_new, truth
            )["dpd45_advance_rate"],
            "final_union_users": len(final_discovery),
            "final_union_dpd45_user_rate": _outcome(final_discovery, truth)[
                "dpd45_user_rate"
            ],
            "final_union_dpd45_advance_rate": _outcome(final_discovery, truth)[
                "dpd45_advance_rate"
            ],
            "state_a_final_union_users": int(len(state_discovery)),
        },
        "plug": {
            "candidate_keys": int(len(stats)),
            "candidate_facts": stats.to_dict("records"),
            "burned_keys": int(len(burned)),
            "state_a": state_plug,
            "holdout": holdout_plug,
            "top_burned_keys": burned[
                ["entity_type", "entity_value", "dpd45_precision", "coverage", "support"]
            ]
            .head(50)
            .to_dict("records"),
        },
        "paths": {
            "markdown": str(paths.markdown),
            "json": str(paths.json),
        },
    }
    paths.json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    paths.markdown.write_text(
        _render_markdown(
            config,
            paths,
            scenario_rows,
            selected_rows,
            excluded_rows,
            graph_status_counts,
            review_graph_net_new,
            final_discovery,
            selected_graph_net_new,
            truth,
            stats,
            burned,
            state_rows,
            holdout_rows,
        ),
        encoding="utf-8",
    )
    return payload


def _load_inputs(store: Path) -> pd.DataFrame:
    with duckdb.connect(str(store), read_only=True) as con:
        return con.execute("SELECT * FROM advances").df()


def _asof_advances(
    advances: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    return advances[
        pd.to_datetime(advances["feature_as_of_ts"]) <= pd.Timestamp(cutoff)
    ].copy()


def _user_truth(advances: pd.DataFrame) -> pd.DataFrame:
    flags = assign(advances)
    truth = pd.DataFrame(
        {
            "user_id": advances.user_id.astype(str),
            "mature_d45": advances.label_mature_d45.fillna(False).astype(bool),
            "dpd45": (
                advances.label_mature_d45.fillna(False).astype(bool)
                & advances.label_gross_dpd45.fillna(False).astype(bool)
            ),
            "is_fraud": advances.is_fraud.fillna(False).astype(bool),
        }
    )
    for scenario in SCENARIOS:
        truth[f"scenario_{scenario.name}"] = flags[f"scenario_{scenario.name}"].fillna(
            False
        ).astype(bool)
    truth["scenario_any"] = flags.scenario_any.fillna(False).astype(bool)

    user_truth = truth.groupby("user_id").agg(
        mature_d45=("mature_d45", "max"),
        dpd45=("dpd45", "max"),
        is_fraud=("is_fraud", "max"),
        scenario_any=("scenario_any", "max"),
        n_advances=("user_id", "size"),
        n_mature_advances=("mature_d45", "sum"),
        n_dpd45_advances=("dpd45", "sum"),
    )
    for scenario in SCENARIOS:
        col = f"scenario_{scenario.name}"
        user_truth[col] = truth.groupby("user_id")[col].max()
    return user_truth


def _scenario_sets(truth: pd.DataFrame) -> dict[str, set[str]]:
    return {
        scenario.name: set(truth.index[truth[f"scenario_{scenario.name}"]])
        for scenario in SCENARIOS
    }


def _scenario_candidates(scenarios: dict[str, set[str]]) -> list[DiscoveryCandidate]:
    candidates = []
    for scenario in SCENARIOS:
        metadata = ScenarioMethod(scenario.name).metadata
        candidates.append(
            DiscoveryCandidate(
                name=metadata.name,
                users=scenarios[scenario.name],
                metadata=metadata,
            )
        )
    return candidates


def _default_graph_discovery() -> Neo4jGraphDiscovery:
    return Neo4jGraphDiscovery(Neo4jClient.from_env())


def _graph_screen_candidates(
    graph_discovery: Neo4jGraphDiscovery,
    scenarios: dict[str, set[str]],
    as_of: pd.Timestamp | None = None,
) -> list[DiscoveryCandidate]:
    return graph_discovery.run(sorted(scenarios), as_of=as_of)


def _select_graph_screens(
    methods: list[DiscoveryCandidate],
    scenario_union: set[str],
    truth: pd.DataFrame,
    min_marginal_users: int,
    min_marginal_dpd45_user_rate: float,
) -> tuple[list[SelectionRow], list[SelectionRow]]:
    result = select_candidates(
        methods,
        baseline_users=scenario_union,
        outcome_fn=lambda users: _outcome(users, truth),
        rule=SelectionRule(
            min_marginal_users=min_marginal_users,
            min_marginal_dpd45_user_rate=min_marginal_dpd45_user_rate,
        ),
    )
    return result.selected, result.excluded


def _outcome(users: frozenset[str] | set[str], truth: pd.DataFrame) -> dict:
    user_ids = {str(user_id) for user_id in users}
    unknown_users = user_ids - {str(user_id) for user_id in truth.index}
    if unknown_users:
        raise ValueError(
            "Outcome users are missing from truth frame: "
            + ", ".join(sorted(unknown_users)[:10])
        )
    if not user_ids:
        return {
            "users": 0,
            "dpd45_users": 0,
            "dpd45_user_rate": 0.0,
            "advances": 0,
            "mature_advances": 0,
            "dpd45_advances": 0,
            "dpd45_advance_rate": 0.0,
        }
    sub_users = truth.loc[truth.index.isin(user_ids)]
    dpd45_users = int(sub_users.dpd45.sum())
    advances = int(sub_users.n_advances.sum())
    mature_advances = int(sub_users.n_mature_advances.sum())
    dpd45_advances = int(sub_users.n_dpd45_advances.sum())
    return {
        "users": len(user_ids),
        "dpd45_users": dpd45_users,
        "dpd45_user_rate": dpd45_users / len(user_ids),
        "advances": advances,
        "mature_advances": mature_advances,
        "dpd45_advances": dpd45_advances,
        "dpd45_advance_rate": dpd45_advances / mature_advances if mature_advances else 0.0,
    }


def _scenario_rows(
    scenarios: list[DiscoveryCandidate],
    scenario_union: set[str],
    truth: pd.DataFrame,
) -> list[dict]:
    rows = []
    for scenario in scenarios:
        scenario_name = str(scenario.metadata.params["scenario_name"])
        outcomes = _outcome(scenario.users, truth)
        row = {
            "scenario": scenario_name,
            "scenario method": scenario.name,
            "method version": scenario.metadata.version,
            "method type": scenario.metadata.method_type,
            "time semantics": scenario.metadata.time_semantics,
            "promotion tier": scenario.metadata.promotion_tier,
            "enforcement projection": scenario.metadata.enforcement_projection,
        }
        row.update(_method_outcome_columns("all discovered", outcomes))
        rows.append(row)
    outcomes = _outcome(scenario_union, truth)
    union_row = {
        "scenario": "scenario union (deduped)",
        "scenario method": "scenario:union",
        "method version": SCENARIOS_VERSION,
        "method type": "scenario",
        "time semantics": "production_safe",
        "promotion tier": "plug_candidate",
        "enforcement projection": "scenario_rule",
    }
    union_row.update(_method_outcome_columns("all discovered", outcomes))
    rows.append(union_row)
    return rows


def _graph_row(method: SelectionRow) -> dict:
    row = {
        "graph method": method.name,
        "status": _graph_status(method),
        "display name": str(method.metadata.params["display_name"]),
        "method version": method.metadata.version,
        "method type": method.metadata.method_type,
        "time semantics": method.metadata.time_semantics,
        "promotion tier": method.metadata.promotion_tier,
        "enforcement projection": method.metadata.enforcement_projection,
        "selected?": "yes" if method.selected else "no",
        "reason": method.reason,
    }
    row.update(_method_outcome_columns("all discovered", method.total))
    row.update(_method_outcome_columns("net-new", method.net))
    row.update(_method_outcome_columns("marginal", method.marginal))
    return row


def _method_outcome_columns(prefix: str, outcome: dict) -> dict:
    if prefix == "all discovered":
        users_key = "all discovered users"
    elif prefix == "net-new":
        users_key = "net-new users beyond scenarios"
    elif prefix == "marginal":
        users_key = "marginal users after dedupe"
    else:
        raise ValueError(f"Unknown method outcome prefix: {prefix!r}")
    return {
        users_key: f"{outcome['users']:,}",
        f"{prefix} DPD45 users/rate": (
            f"{outcome['dpd45_users']:,}/{outcome['users']:,} "
            f"({_pct(outcome['dpd45_user_rate'])})"
        ),
        f"{prefix} DPD45 advances/rate": (
            f"{outcome['dpd45_advances']:,}/{outcome['mature_advances']:,} "
            f"({_pct(outcome['dpd45_advance_rate'])})"
        ),
    }


def _graph_status(method: SelectionRow) -> str:
    if method.selected:
        return "promoted_to_plug_derivation"
    if not method.metadata.plug_eligible:
        return "review_only"
    if method.reason == "min_marginal_users":
        return "below_min_marginal_users"
    if method.reason == "min_marginal_dpd45_user_rate":
        return "below_min_marginal_dpd45_user_rate"
    raise ValueError(f"Unknown graph selection reason: {method.reason!r}")


def _plug_bucket_row(bucket: str, report_bucket: dict) -> dict:
    outcomes = report_bucket["outcomes"]
    return {
        "bucket": bucket,
        "users": f"{report_bucket['n_users']:,}",
        "DPD45 users": f"{outcomes['n_dpd45_users']:,}/{outcomes['n_users_with_advances']:,}",
        "DPD45 user rate": _pct(outcomes["dpd45_user_rate"]),
        "advances": f"{outcomes['n_advances']:,}",
        "DPD45 advances": (
            f"{outcomes['n_dpd45_advances']:,}/{outcomes['n_mature_advances']:,}"
        ),
        "DPD45 advance rate": _pct(outcomes["dpd45_advance_rate"]),
    }


def _render_markdown(
    config: ControlLoopReportConfig,
    paths: ReportPaths,
    scenario_rows: list[dict],
    selected_rows: list[dict],
    excluded_rows: list[dict],
    graph_status_counts: dict[str, int],
    review_graph_net_new: set[str],
    final_discovery: set[str],
    selected_graph_net_new: set[str],
    truth: pd.DataFrame,
    stats: pd.DataFrame,
    burned: pd.DataFrame,
    state_rows: list[dict],
    holdout_rows: list[dict],
) -> str:
    final_outcomes = _outcome(final_discovery, truth)
    graph_outcomes = _outcome(selected_graph_net_new, truth)
    review_graph_outcomes = _outcome(review_graph_net_new, truth)
    displayed_statuses = ", ".join(sorted(config.include_statuses))
    scenario_headers = [
        "scenario",
        "scenario method",
        "method version",
        "method type",
        "time semantics",
        "promotion tier",
        "enforcement projection",
        "all discovered users",
        "all discovered DPD45 users/rate",
        "all discovered DPD45 advances/rate",
    ]
    graph_headers = [
        "graph method",
        "status",
        "display name",
        "method version",
        "method type",
        "time semantics",
        "promotion tier",
        "enforcement projection",
        "all discovered users",
        "all discovered DPD45 users/rate",
        "all discovered DPD45 advances/rate",
        "net-new users beyond scenarios",
        "net-new DPD45 users/rate",
        "net-new DPD45 advances/rate",
        "selected?",
        "reason",
    ]
    return f"""# Fraud Control Loop Report

This report is the full sample control-loop view: scenarios, graph review screens, promotion into plug derivation, plug validation, State A, and holdout. All unions are deduped by `user_id`.

Graph selection rule for this run: include a graph method only when its metadata is plug-eligible (`promotion_tier=plug_candidate`, `time_semantics` is `leakfree_asof` or `production_safe`, and `enforcement_projection` is not `none`) and its marginal net-new contribution after scenario union and already-selected graph methods has at least `{config.graph_min_marginal_users}` users and DPD45 user rate at or above `{_pct(config.graph_min_marginal_dpd45_user_rate)}`. Snapshot-review, low-precision, or duplicate graph methods are shown for auditability but excluded from final discovery and plug derivation.

Displayed graph statuses: `{displayed_statuses}`.

## Scenario Method Screen

{_table(scenario_rows, scenario_headers)}

## Graph Method Screen

Status counts before display filtering:

- promoted_to_plug_derivation: `{graph_status_counts["promoted_to_plug_derivation"]}`
- review_only: `{graph_status_counts["review_only"]}`
- below_min_marginal_users: `{graph_status_counts["below_min_marginal_users"]}`
- below_min_marginal_dpd45_user_rate: `{graph_status_counts["below_min_marginal_dpd45_user_rate"]}`

Review-only graph net-new users beyond scenarios: `{len(review_graph_net_new):,}`. User-level DPD45: `{review_graph_outcomes["dpd45_users"]:,}/{review_graph_outcomes["users"]:,}` at `{_pct(review_graph_outcomes["dpd45_user_rate"])}`. Advance-level DPD45: `{review_graph_outcomes["dpd45_advances"]:,}/{review_graph_outcomes["mature_advances"]:,}` at `{_pct(review_graph_outcomes["dpd45_advance_rate"])}`. These are visible discovery leads, not plug-derived users, until their method metadata is upgraded to leak-free/as-of or production-safe.

### Promoted Graph Methods

{_table(selected_rows, graph_headers) if selected_rows else "No graph methods passed the marginal selection rule or display filter."}

### Graph Review / Exclusion Rows

{_table(excluded_rows, graph_headers) if excluded_rows else "No graph methods matched the display filter."}

## Discovery Summary

- Scenario union users: `{_scenario_union_users(scenario_rows)}`
- Selected graph net-new users beyond scenarios: `{len(selected_graph_net_new):,}`
- Selected graph net-new DPD45 user rate: `{graph_outcomes["dpd45_users"]:,}/{graph_outcomes["users"]:,}` at `{_pct(graph_outcomes["dpd45_user_rate"])}`
- Selected graph net-new DPD45 advance rate: `{graph_outcomes["dpd45_advances"]:,}/{graph_outcomes["mature_advances"]:,}` at `{_pct(graph_outcomes["dpd45_advance_rate"])}`
- Final discovery union users: `{final_outcomes["users"]:,}`
- Final discovery union DPD45 user rate: `{_pct(final_outcomes["dpd45_user_rate"])}`
- Final discovery union DPD45 advance rate: `{_pct(final_outcomes["dpd45_advance_rate"])}`

## Plug-Hole Validation From Final Discovery Union

Default plug gates: support >= `{config.plug_config.min_support}`, discovery coverage >= `{config.plug_config.min_coverage}`, DPD45 precision >= `{config.plug_config.block_tier_precision}`.

- Candidate keys from State A final discovery: `{len(stats):,}`
- Qualified burned keys: `{len(burned):,}`

### State A Backtest

{_table(state_rows, ["bucket", "users", "DPD45 users", "DPD45 user rate", "advances", "DPD45 advances", "DPD45 advance rate"])}

### Holdout Cross-Validation

{_table(holdout_rows, ["bucket", "users", "DPD45 users", "DPD45 user rate", "advances", "DPD45 advances", "DPD45 advance rate"])}

## Notes

- Yes, every union above is deduped by `user_id`.
- Review-only graph methods are shown but not included in the final plug union. Methods that duplicate already-selected graph users are also excluded if their marginal contribution fails the bar.
- Before production blocking, a graph method needs a leak-free as-of or production-safe implementation if it uses hindsight-like seeds or entity risk summaries.
- Plug validation explicitly separates discovery coverage (`covered_discovery` and `uncovered_discovery`) from actual performance on users touched outside the discovery set (`outside_discovery`).

Machine-readable JSON: `{paths.json}`
"""


def _scenario_union_users(scenario_rows: list[dict]) -> str:
    return scenario_rows[-1]["all discovered users"]


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _table(rows: list[dict], headers: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[header]) for header in headers) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--refresh-key", default=DEFAULT_REFRESH_KEY)
    parser.add_argument("--graph-min-marginal-users", type=int, default=10)
    parser.add_argument("--graph-min-marginal-dpd45-user-rate", type=float, default=0.50)
    parser.add_argument("--neo4j-uri", default=None)
    parser.add_argument("--neo4j-user", default=None)
    parser.add_argument("--neo4j-password", default=None)
    parser.add_argument("--neo4j-database", default=None)
    parser.add_argument(
        "--include-status",
        action="append",
        choices=["all", *sorted(GRAPH_STATUSES)],
        help=(
            "Graph row status to show; repeat to include multiple statuses. "
            "Default: all."
        ),
    )
    parser.add_argument("--min-support", type=int, default=ControlConfig().min_support)
    parser.add_argument("--min-coverage", type=int, default=ControlConfig().min_coverage)
    parser.add_argument(
        "--block-tier-precision",
        type=float,
        default=ControlConfig().block_tier_precision,
    )
    args = parser.parse_args(argv)
    include_statuses = (
        GRAPH_STATUSES
        if not args.include_status or "all" in args.include_status
        else frozenset(args.include_status)
    )
    client = Neo4jClient.from_env(
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_password,
        database=args.neo4j_database,
    )
    try:
        report = generate_control_loop_report(
            ControlLoopReportConfig(
                store=args.store,
                out_dir=args.out_dir,
                refresh_key=args.refresh_key,
                graph_min_marginal_users=args.graph_min_marginal_users,
                graph_min_marginal_dpd45_user_rate=args.graph_min_marginal_dpd45_user_rate,
                plug_config=ControlConfig(
                    min_support=args.min_support,
                    min_coverage=args.min_coverage,
                    block_tier_precision=args.block_tier_precision,
                ),
                include_statuses=include_statuses,
            ),
            graph_discovery=Neo4jGraphDiscovery(client),
        )
    finally:
        client.close()
    print(
        json.dumps(
            {
                "markdown_report": report["paths"]["markdown"],
                "json_report": report["paths"]["json"],
                "final_union_users": report["final_discovery"]["final_union_users"],
                "review_graph_net_new_users": report["review_graph_net_new_users"],
                "graph_status_counts": report["graph_status_counts"],
                "burned_keys": report["plug"]["burned_keys"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
