"""AutoML command line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from automl.cli import data, eval, experiment, project, trial, validate


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
    data.add_parser(subparsers)
    eval.add_parser(subparsers)
    validate.add_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


__all__ = ["build_parser", "main"]
