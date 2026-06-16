"""Ad-hoc discovery evaluator — score one candidate Cypher in the report's format.

Talk out a discovery idea, run it as a single Cypher query that returns ``user_id``
rows, and get back the standard analysis panel (DPD45 user-rate + advance-rate,
net-new beyond the current discovery union, per-method overlap) WITHOUT editing the
method catalog or re-running the full control-loop report.

Net-new/overlap read the sidecar cache written by ``control_loop_report`` (per-method
+ union user-id sets). With no cache, the candidate's own DPD45 panel still prints;
net-new is reported as unavailable. Promotion of a proven query into the catalog is a
separate, later, reviewed step (see the README "Adding a graph discovery pattern").
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from projects.fraud_anomaly_detection.neo4j_codex.control.control_loop_report import (
    DEFAULT_OUT_DIR,
    DEFAULT_REFRESH_KEY,
    DEFAULT_STORE,
)
from projects.fraud_anomaly_detection.neo4j_codex.control.discovery import metrics
from projects.fraud_anomaly_detection.neo4j_codex.control.graph.client import (
    GraphQueryRunner,
    Neo4jClient,
)


def run_candidate_users(
    runner: GraphQueryRunner,
    cypher: str,
    params: dict | None = None,
) -> set[str]:
    """Run a candidate Cypher and collect its ``user_id`` rows."""
    rows = runner.run(cypher, params or {})
    users: set[str] = set()
    for row in rows:
        if "user_id" not in row:
            raise ValueError("candidate query must RETURN a user_id column")
        users.add(str(row["user_id"]))
    return users


def evaluate_candidate(
    *,
    users: set[str],
    store: Path,
    cache: dict | None,
    name: str = "candidate",
) -> dict:
    """Score a candidate user set in the control-loop report's analysis format.

    Uses the shared ``metrics.outcome`` definition (DPD45 user-rate over the whole
    set), so the numbers line up with the report's scenario/graph rows. Candidate
    users absent from the store (no advance row, e.g. link-only) cannot be scored;
    they are reported as ``n_users_off_store`` and excluded from the panels.
    """
    truth = metrics.user_truth(metrics.load_advances(store))
    known = {str(u) for u in truth.index}
    candidate = {str(u) for u in users}
    scored = candidate & known
    result: dict = {
        "name": name,
        "store": str(store),
        "n_candidate_users": len(candidate),
        "n_users_off_store": len(candidate - known),
        "candidate": metrics.outcome(scored, truth),
    }
    if cache is None:
        result["net_new"] = None
        result["per_method"] = None
        result["cache"] = "unavailable — run control_loop_report to populate the cache"
        return result

    final_discovery = {str(u) for u in cache.get("final_discovery", [])}
    net_new_users = scored - final_discovery
    result["net_new"] = {
        "n_net_new_users": len(net_new_users),
        "outcomes": metrics.outcome(net_new_users, truth),
    }
    result["per_method"] = [
        {
            "method": method,
            "overlap_users": len(candidate & set(method_users)),
            "net_new_beyond_method": len(candidate - set(method_users)),
        }
        for method, method_users in sorted(cache.get("methods", {}).items())
    ]
    result["cache"] = {
        "store": cache.get("store"),
        "scenario_version": cache.get("scenario_version"),
        "stale": cache.get("store") != str(store),
    }
    return result


def load_cache(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _outcome_lines(label: str, o: dict) -> list[str]:
    prefix = f"{label} " if label else ""
    return [
        (
            f"- {prefix}DPD45 user rate: `{o['dpd45_users']:,}/{o['users']:,}` "
            f"(`{_pct(o['dpd45_user_rate'])}`)"
        ),
        (
            f"- {prefix}DPD45 advance rate: `{o['dpd45_advances']:,}/"
            f"{o['mature_advances']:,}` (`{_pct(o['dpd45_advance_rate'])}`)"
        ),
    ]


def render_markdown(result: dict) -> str:
    panel = result["candidate"]
    off_store = (
        f" (`{result['n_users_off_store']:,}` off-store, not scored)"
        if result["n_users_off_store"]
        else ""
    )
    lines = [
        f"# Ad-hoc discovery candidate: `{result['name']}`",
        "",
        f"Store: `{result['store']}`",
        "",
        "## Candidate facts",
        f"- Users: `{result['n_candidate_users']:,}`{off_store}",
        *_outcome_lines("", panel),
        "",
    ]
    net_new = result["net_new"]
    if net_new is None:
        lines += ["## Net-new beyond discovery union", "", f"_{result['cache']}_", ""]
    else:
        stale = " (stale — cache store differs)" if result["cache"]["stale"] else ""
        lines += [
            f"## Net-new beyond final discovery union{stale}",
            f"- Net-new users: `{net_new['n_net_new_users']:,}`",
            *_outcome_lines("Net-new", net_new["outcomes"]),
            "",
            "## Overlap per cached method",
            "| method | overlap users | net-new beyond method |",
            "| --- | --- | --- |",
        ]
        for row in result["per_method"]:
            lines.append(
                f"| {row['method']} | {row['overlap_users']:,} | "
                f"{row['net_new_beyond_method']:,} |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--cypher", help="candidate Cypher returning a user_id column")
    src.add_argument("--cypher-file", type=Path, help="path to a file with the Cypher")
    parser.add_argument("--params", default="{}", help="JSON object of query params")
    parser.add_argument("--name", default="candidate")
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_OUT_DIR / f"{DEFAULT_REFRESH_KEY}.cache.json",
        help="discovery cache sidecar written by control_loop_report",
    )
    parser.add_argument("--neo4j-uri", default=None)
    parser.add_argument("--neo4j-user", default=None)
    parser.add_argument("--neo4j-password", default=None)
    parser.add_argument("--neo4j-database", default=None)
    parser.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    args = parser.parse_args(argv)

    cypher = args.cypher or args.cypher_file.read_text(encoding="utf-8")
    params = json.loads(args.params)
    client = Neo4jClient.from_env(
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_password,
        database=args.neo4j_database,
    )
    try:
        users = run_candidate_users(client, cypher, params)
    finally:
        client.close()
    result = evaluate_candidate(
        users=users,
        store=args.store,
        cache=load_cache(args.cache),
        name=args.name,
    )
    print(json.dumps(result, indent=2) if args.json else render_markdown(result))


if __name__ == "__main__":
    main()
