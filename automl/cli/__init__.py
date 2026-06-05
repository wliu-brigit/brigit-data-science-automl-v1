"""AutoML command line entry point."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from automl.cli import data, eval, experiment, mlflow, project, trial, validate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="automl")
    parser.add_argument("--project")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--namespace", default="")
    parser.add_argument("--experiment-id")
    subparsers = parser.add_subparsers(dest="noun", required=True)
    project.add_parser(subparsers)
    experiment.add_parser(subparsers)
    trial.add_parser(subparsers)
    mlflow.add_parser(subparsers)
    data.add_parser(subparsers)
    eval.add_parser(subparsers)
    validate.add_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.monotonic()
    exit_code = 1
    try:
        exit_code = int(args.func(args))
        return exit_code
    finally:
        # Inside an agent session every CLI verb is one loop step; record its
        # wall-clock on the session timeline so the published timing covers
        # the whole loop, not just the proposer/coder agent spans. No-op (and
        # never raises) outside an agent session.
        from automl.agent.timeline.steps import record_cli_step

        record_cli_step(
            _step_name(args),
            duration_s=time.monotonic() - started,
            exit_code=exit_code,
        )


def _step_name(args: argparse.Namespace) -> str:
    """``"<noun> <verb>"`` from parsed args (e.g. ``"experiment proposer-context"``)."""
    noun = str(getattr(args, "noun", "") or "")
    verb = str(getattr(args, "action", "") or getattr(args, "target", "") or "")
    return f"{noun} {verb}".strip()


__all__ = ["build_parser", "main"]
