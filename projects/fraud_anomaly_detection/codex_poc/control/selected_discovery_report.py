"""Repeatable selected-discovery report for scenarios, graph screens, and plugs."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

from projects.fraud_anomaly_detection.codex_poc.control import plug
from projects.fraud_anomaly_detection.codex_poc.control.config import ControlConfig
from projects.fraud_anomaly_detection.codex_poc.control.discovery.metadata import (
    MethodMetadata,
)
from projects.fraud_anomaly_detection.codex_poc.control.discovery.selection import (
    DiscoveryCandidate,
    SelectionRule,
    SelectionRow,
    select_candidates,
)
from projects.fraud_anomaly_detection.codex_poc.control.holdout import two_state_split
from projects.fraud_anomaly_detection.codex_poc.control.plug_report import summarize_plugs
from projects.fraud_anomaly_detection.graph.discover import (
    bad_neighbours,
    residual_ring_members,
    suspicion_queue,
)
from projects.fraud_anomaly_detection.graph.load import load_graph
from projects.fraud_anomaly_detection.scenarios import SCENARIOS, SCENARIOS_VERSION, assign

DEFAULT_STORE = Path("projects/fraud_anomaly_detection/data/graph/fraud_graph.duckdb")
DEFAULT_OUT_DIR = Path("projects/fraud_anomaly_detection/codex_poc/reports")
DEFAULT_REFRESH_KEY = "selected_discovery_plug_report"


@dataclass(frozen=True)
class SelectedReportConfig:
    """Parameters for the report generation and graph-method selection."""

    store: Path = DEFAULT_STORE
    out_dir: Path = DEFAULT_OUT_DIR
    refresh_key: str = DEFAULT_REFRESH_KEY
    graph_min_marginal_users: int = 10
    graph_min_marginal_dpd45_user_rate: float = 0.50
    plug_config: ControlConfig = field(default_factory=ControlConfig)


@dataclass(frozen=True)
class ReportPaths:
    markdown: Path
    json: Path


def generate_selected_discovery_report(config: SelectedReportConfig) -> dict:
    """Generate the selected discovery + plug report and write Markdown/JSON files."""
    if not config.store.exists():
        raise FileNotFoundError(f"store not found: {config.store}")

    config.out_dir.mkdir(parents=True, exist_ok=True)
    paths = ReportPaths(
        markdown=config.out_dir / f"{config.refresh_key}.md",
        json=config.out_dir / f"{config.refresh_key}.json",
    )

    advances, edges = _load_inputs(config.store)
    truth = _user_truth(advances)
    scenarios = _scenario_sets(truth)
    scenario_union = set().union(*scenarios.values()) if scenarios else set()
    graph_methods = _graph_method_sets(config.store, advances, edges, truth, scenarios)
    selected_graphs, excluded_graphs = _select_graph_methods(
        graph_methods,
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
    state_discovery = pd.Series(
        sorted(final_discovery & set(split.state_a_users)),
        dtype="string",
    )
    holdout_discovery = pd.Series(
        sorted(final_discovery & set(split.holdout_users)),
        dtype="string",
    )
    stats = plug.candidate_stats(config.store, state_discovery, eligible_users=split.state_a_users)
    burned = plug.qualify(stats, config.plug_config)
    state_plug = summarize_plugs(
        config.store,
        burned,
        discovery_users=state_discovery,
        eligible_users=split.state_a_users,
    )
    holdout_plug = summarize_plugs(
        config.store,
        burned,
        discovery_users=holdout_discovery,
        eligible_users=split.holdout_users,
        start_ts=split.cutoff,
    )

    scenario_rows = _scenario_rows(scenarios, scenario_union, truth)
    selected_rows = [_graph_row(method) for method in selected_graphs]
    excluded_rows = [_graph_row(method) for method in excluded_graphs]
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
        },
        "selection_rule": {
            "min_marginal_users": config.graph_min_marginal_users,
            "min_marginal_dpd45_user_rate": config.graph_min_marginal_dpd45_user_rate,
        },
        "scenario_rows": scenario_rows,
        "selected_graph_rows": selected_rows,
        "excluded_graph_rows": excluded_rows,
        "final_discovery": {
            "scenario_union_users": len(scenario_union),
            "selected_graph_net_new_users": len(selected_graph_net_new),
            "selected_graph_net_new_dpd45_user_rate": _outcome(
                selected_graph_net_new, truth
            )["dpd45_user_rate"],
            "final_union_users": len(final_discovery),
            "final_union_dpd45_user_rate": _outcome(final_discovery, truth)[
                "dpd45_user_rate"
            ],
            "final_union_dpd45_advance_rate": _outcome(final_discovery, truth)[
                "dpd45_advance_rate"
            ],
        },
        "plug": {
            "candidate_keys": int(len(stats)),
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


def _load_inputs(store: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with duckdb.connect(str(store), read_only=True) as con:
        advances = con.execute("SELECT * FROM advances").df()
        edges = con.execute(
            """
            SELECT DISTINCT
                CAST(user_id AS VARCHAR) AS user_id,
                CAST(entity_type AS VARCHAR) AS entity_type,
                CAST(entity_value AS VARCHAR) AS entity_value
            FROM edges
            """
        ).df()
    return advances, edges


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


def _graph_method_sets(
    store: Path,
    advances: pd.DataFrame,
    edges: pd.DataFrame,
    truth: pd.DataFrame,
    scenarios: dict[str, set[str]],
) -> list[DiscoveryCandidate]:
    residual_users = set(truth.index[~truth.scenario_any & ~truth.is_fraud])
    g_scen = load_graph(store, base=advances, node_attrs=("is_fraud",), scenarios=True)
    g_no_scen = load_graph(store, base=advances, node_attrs=("is_fraud",), scenarios=False)

    methods = [
        DiscoveryCandidate(
            name="residual_ring_members",
            users=set(residual_ring_members(g_scen, flag="scenario_any").user_id.astype(str)),
            metadata=_graph_metadata("residual_ring_members", promotion_tier="review_queue"),
        ),
        DiscoveryCandidate(
            name="suspicion_queue_top200",
            users=set(
                suspicion_queue(
                    g_scen,
                    seed_flag="is_fraud",
                    exclude_flags=("scenario_any", "is_fraud"),
                    top_n=200,
                ).user_id.astype(str)
            ),
            metadata=_graph_metadata("suspicion_queue_top200", promotion_tier="review_queue"),
        ),
        DiscoveryCandidate(
            name="fraud_neighbours_hops2",
            users=set(bad_neighbours(g_no_scen, flags=("is_fraud",), max_hops=2).user_id.astype(str)),
            metadata=_graph_metadata("fraud_neighbours_hops2", promotion_tier="review_queue"),
        ),
        DiscoveryCandidate(
            name="high_risk_entity_members_scenario_fraud_seed",
            users=_high_risk_entity_members(edges, truth, residual_users),
            metadata=_graph_metadata(
                "high_risk_entity_members_scenario_fraud_seed",
                promotion_tier="plug_candidate",
            ),
        ),
        DiscoveryCandidate(
            name="multi_witness_neighbors_scenario_fraud_seed",
            users=_multi_witness_neighbors(edges, truth, residual_users),
            metadata=_graph_metadata(
                "multi_witness_neighbors_scenario_fraud_seed",
                promotion_tier="review_queue",
            ),
        ),
    ]
    for scenario_name, scenario_users in scenarios.items():
        methods.append(
            DiscoveryCandidate(
                name=f"scenario_neighborhood:{scenario_name}",
                users=_scenario_neighborhood(edges, residual_users, scenario_users),
                metadata=_graph_metadata(
                    f"scenario_neighborhood:{scenario_name}",
                    promotion_tier="review_queue",
                    params={"scenario_name": scenario_name},
                ),
            )
        )

    return [
        DiscoveryCandidate(
            name=method.name,
            users={str(user_id) for user_id in method.users},
            metadata=method.metadata,
        )
        for method in methods
    ]


def _graph_metadata(
    name: str,
    *,
    promotion_tier: str,
    params: dict[str, object] | None = None,
) -> MethodMetadata:
    return MethodMetadata(
        name=f"graph:{name}",
        version="selected-report-1",
        method_type="graph",
        time_semantics="snapshot_review",
        promotion_tier=promotion_tier,  # type: ignore[arg-type]
        enforcement_projection="entity_key",
        params={"source": "selected_discovery_report", **(params or {})},
    )


def _scenario_neighborhood(
    edges: pd.DataFrame,
    residual_users: set[str],
    scenario_users: set[str],
) -> set[str]:
    seed_edges = edges[edges.user_id.isin(scenario_users)]
    candidate_edges = edges[edges.user_id.isin(residual_users)]
    joined = candidate_edges.merge(
        seed_edges.rename(columns={"user_id": "seed_user"}),
        on=["entity_type", "entity_value"],
        how="inner",
    )
    return set(joined.user_id.astype(str))


def _high_risk_entity_members(
    edges: pd.DataFrame,
    truth: pd.DataFrame,
    residual_users: set[str],
) -> set[str]:
    frame = edges.merge(
        truth[["scenario_any", "is_fraud"]],
        left_on="user_id",
        right_index=True,
        how="left",
    ).fillna(False)
    stats = frame.groupby(["entity_type", "entity_value"]).agg(
        entity_users=("user_id", "nunique"),
        fraud_users=("is_fraud", "sum"),
        scenario_users=("scenario_any", "sum"),
    ).reset_index()
    risky = stats[
        (stats.entity_users >= 3)
        & (stats.entity_users <= 50)
        & ((stats.fraud_users >= 1) | (stats.scenario_users >= 2))
    ]
    candidates = edges[edges.user_id.isin(residual_users)].merge(
        risky[["entity_type", "entity_value"]],
        on=["entity_type", "entity_value"],
        how="inner",
    )
    return set(candidates.user_id.astype(str))


def _multi_witness_neighbors(
    edges: pd.DataFrame,
    truth: pd.DataFrame,
    residual_users: set[str],
) -> set[str]:
    risky_users = set(truth.index[truth.scenario_any | truth.is_fraud])
    joined = edges[edges.user_id.isin(residual_users)].merge(
        edges[edges.user_id.isin(risky_users)].rename(columns={"user_id": "seed_user"}),
        on=["entity_type", "entity_value"],
        how="inner",
    )
    grouped = joined.groupby("user_id").agg(
        shared_type_count=("entity_type", "nunique")
    ).reset_index()
    return set(grouped.loc[grouped.shared_type_count >= 2, "user_id"].astype(str))


def _select_graph_methods(
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


def _outcome(users: set[str], truth: pd.DataFrame) -> dict:
    user_ids = {str(user_id) for user_id in users}
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
    scenarios: dict[str, set[str]],
    scenario_union: set[str],
    truth: pd.DataFrame,
) -> list[dict]:
    rows = []
    for scenario in SCENARIOS:
        outcomes = _outcome(scenarios[scenario.name], truth)
        rows.append(
            {
                "scenario": scenario.name,
                "users found": f"{outcomes['users']:,}",
                "DPD45 user rate": _pct(outcomes["dpd45_user_rate"]),
                "DPD45 advances": f"{outcomes['dpd45_advances']:,}/{outcomes['mature_advances']:,}",
                "DPD45 advance rate": _pct(outcomes["dpd45_advance_rate"]),
            }
        )
    outcomes = _outcome(scenario_union, truth)
    rows.append(
        {
            "scenario": "scenario union (deduped)",
            "users found": f"{outcomes['users']:,}",
            "DPD45 user rate": _pct(outcomes["dpd45_user_rate"]),
            "DPD45 advances": f"{outcomes['dpd45_advances']:,}/{outcomes['mature_advances']:,}",
            "DPD45 advance rate": _pct(outcomes["dpd45_advance_rate"]),
        }
    )
    return rows


def _graph_row(method: SelectionRow) -> dict:
    return {
        "graph method": method.name,
        "total users / DPD45": (
            f"{method.total['users']:,} / {_pct(method.total['dpd45_user_rate'])}"
        ),
        "net-new beyond scenarios / DPD45": (
            f"{method.net['users']:,} / {_pct(method.net['dpd45_user_rate'])}"
        ),
        "marginal after dedupe / DPD45": (
            f"{method.marginal['users']:,} / {_pct(method.marginal['dpd45_user_rate'])}"
        ),
        "selected?": "yes" if method.selected else "no",
        "reason": method.reason,
    }


def _plug_bucket_row(bucket: str, report_bucket: dict) -> dict:
    outcomes = report_bucket["outcomes"]
    return {
        "bucket": bucket,
        "users": f"{report_bucket['n_users']:,}",
        "advances": f"{outcomes['n_advances']:,}",
        "DPD45 advances": (
            f"{outcomes['n_dpd45_advances']:,}/{outcomes['n_mature_advances']:,}"
        ),
        "DPD45 advance rate": _pct(outcomes["dpd45_advance_rate"]),
    }


def _render_markdown(
    config: SelectedReportConfig,
    paths: ReportPaths,
    scenario_rows: list[dict],
    selected_rows: list[dict],
    excluded_rows: list[dict],
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
    return f"""# Selected Discovery + Plug Validation Report

This report uses the requested structure: scenarios first, then graph screens, then only selected graph methods are unioned with the scenarios. All unions are deduped by `user_id`.

Graph selection rule for this run: include a graph method only when its marginal net-new contribution after scenario union and already-selected graph methods has at least `{config.graph_min_marginal_users}` users and DPD45 user rate at or above `{_pct(config.graph_min_marginal_dpd45_user_rate)}`. Low-precision or duplicate graph methods are shown for auditability but excluded from final discovery and plug derivation.

## Scenario Performance

{_table(scenario_rows, ["scenario", "users found", "DPD45 user rate", "DPD45 advances", "DPD45 advance rate"])}

## Graph Method Screen

### Selected Graph Methods

{_table(selected_rows, ["graph method", "total users / DPD45", "net-new beyond scenarios / DPD45", "marginal after dedupe / DPD45", "selected?", "reason"]) if selected_rows else "No graph methods passed the marginal selection rule."}

### Screened But Excluded Graph Methods

{_table(excluded_rows, ["graph method", "total users / DPD45", "net-new beyond scenarios / DPD45", "marginal after dedupe / DPD45", "selected?", "reason"]) if excluded_rows else "No graph methods were excluded."}

## Final Deduped Discovery Union

- Scenario union users: `{_scenario_union_users(scenario_rows)}`
- Selected graph net-new users beyond scenarios: `{len(selected_graph_net_new):,}`
- Selected graph net-new DPD45 user rate: `{_pct(graph_outcomes["dpd45_user_rate"])}`
- Final discovery union users: `{final_outcomes["users"]:,}`
- Final discovery union DPD45 user rate: `{_pct(final_outcomes["dpd45_user_rate"])}`
- Final discovery union DPD45 advance rate: `{_pct(final_outcomes["dpd45_advance_rate"])}`

## Plug-Hole Validation From Final Discovery Union

Default plug gates: support >= `{config.plug_config.min_support}`, discovery coverage >= `{config.plug_config.min_coverage}`, DPD45 precision >= `{config.plug_config.block_tier_precision}`.

- Candidate keys from State A final discovery: `{len(stats):,}`
- Qualified burned keys: `{len(burned):,}`

### State A Backtest

{_table(state_rows, ["bucket", "users", "advances", "DPD45 advances", "DPD45 advance rate"])}

### Holdout Cross-Validation

{_table(holdout_rows, ["bucket", "users", "advances", "DPD45 advances", "DPD45 advance rate"])}

## Notes

- Yes, every union above is deduped by `user_id`.
- Low-precision graph methods are not included in the final union. Methods that duplicate already-selected graph users are also excluded if their marginal contribution fails the bar.
- The selected graph methods here are still discovery methods. Before production blocking, each needs a leak-free as-of implementation if it uses hindsight-like seeds or entity risk summaries.
- Plug validation explicitly separates discovery coverage (`covered_discovery` and `uncovered_discovery`) from actual performance on users touched outside the discovery set (`outside_discovery`).

Machine-readable JSON: `{paths.json}`
"""


def _scenario_union_users(scenario_rows: list[dict]) -> str:
    return scenario_rows[-1]["users found"]


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _table(rows: list[dict], headers: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--refresh-key", default=DEFAULT_REFRESH_KEY)
    parser.add_argument("--graph-min-marginal-users", type=int, default=10)
    parser.add_argument("--graph-min-marginal-dpd45-user-rate", type=float, default=0.50)
    parser.add_argument("--min-support", type=int, default=ControlConfig().min_support)
    parser.add_argument("--min-coverage", type=int, default=ControlConfig().min_coverage)
    parser.add_argument(
        "--block-tier-precision",
        type=float,
        default=ControlConfig().block_tier_precision,
    )
    args = parser.parse_args(argv)
    report = generate_selected_discovery_report(
        SelectedReportConfig(
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
        )
    )
    print(
        json.dumps(
            {
                "markdown_report": report["paths"]["markdown"],
                "json_report": report["paths"]["json"],
                "final_union_users": report["final_discovery"]["final_union_users"],
                "burned_keys": report["plug"]["burned_keys"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
