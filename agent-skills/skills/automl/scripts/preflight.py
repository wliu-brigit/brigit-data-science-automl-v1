from __future__ import annotations

import argparse
import json
import math
import shlex
from pathlib import Path

from automl.agent.run_options import add_experiment_run_options, options_from_namespace


DRY_RUN_ALIASES = {
    "try it out",
    "smoke test",
    "test run",
    "quick test",
}


def _base_payload(
    *,
    mode: str,
    project: str = "",
    dry_run: bool = False,
    namespace: str = "",
    max_iterations: int | None = None,
    time_budget_hours: float | None = None,
    refresh_data: bool = False,
    refresh_source: bool = False,
    user_instructions: list[str] | None = None,
    needs_confirmation: bool = False,
    interpretation: str = "",
    query: str = "",
) -> dict:
    return {
        "mode": mode,
        "project": project,
        "dry_run": dry_run,
        "namespace": namespace,
        "max_iterations": max_iterations,
        "time_budget_hours": time_budget_hours,
        "refresh_data": refresh_data,
        "refresh_source": refresh_source,
        "user_instructions": user_instructions or [],
        "needs_confirmation": needs_confirmation,
        "interpretation": interpretation,
        "query": query,
    }


def _error_payload(message: str) -> dict:
    return {
        "mode": "error",
        "error": message,
        "needs_confirmation": False,
    }


def _run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="automl experiment run", add_help=False)
    parser.add_argument("noun")
    parser.add_argument("action")
    add_experiment_run_options(
        parser,
        include_project_flags=True,
        include_confirmation=True,
    )
    return parser


def parse_arguments(raw: str) -> dict:
    raw = raw.strip()
    if not raw:
        return _base_payload(
            mode="help",
            interpretation="No arguments supplied.",
        )

    lowered = raw.lower()
    if lowered in {"help", "--help", "-h"}:
        return _base_payload(
            mode="help",
            interpretation="Help requested.",
        )

    if lowered in DRY_RUN_ALIASES:
        return _base_payload(
            mode="run",
            dry_run=True,
            max_iterations=3,
            needs_confirmation=True,
            interpretation=f"Interpreted '{raw}' as a dry-run smoke test.",
        )

    try:
        tokens = shlex.split(raw)
    except ValueError as exc:
        return _error_payload(f"Unable to parse arguments: {exc}")

    if len(tokens) >= 2 and tokens[:2] == ["experiment", "run"]:
        parser = _run_parser()
        try:
            parsed, unknown = parser.parse_known_args(tokens)
        except (argparse.ArgumentError, SystemExit) as exc:
            return _error_payload(f"Unable to parse run arguments: {exc}")

        if unknown:
            return _error_payload(f"Unsupported arguments: {' '.join(unknown)}")

        options = options_from_namespace(parsed)
        if options.time_budget is not None and (
            not math.isfinite(options.time_budget) or options.time_budget <= 0
        ):
            return _error_payload("--time-budget must be a finite positive number")

        auto_confirm = bool(options.auto_confirm)
        return _base_payload(
            mode="run",
            project=options.project,
            dry_run=options.dry_run,
            namespace=options.namespace,
            max_iterations=options.max_iter,
            time_budget_hours=options.time_budget,
            refresh_data=options.refresh_data,
            refresh_source=options.refresh_source,
            user_instructions=list(options.instructions),
            needs_confirmation=not auto_confirm,
            interpretation="Parsed canonical automl experiment run invocation.",
        )

    return _base_payload(
        mode="investigate",
        interpretation="Treating arguments as a read-only AutoML investigation request.",
        query=raw,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--arguments", default="")
    args = parser.parse_args()

    payload = parse_arguments(args.arguments)
    payload["project_root"] = str(args.project_root)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
