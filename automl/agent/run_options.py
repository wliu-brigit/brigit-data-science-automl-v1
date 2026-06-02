"""Internal option helpers for the experiment agent loop."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentRunOptions:
    project: str = ""
    dry_run: bool = False
    namespace: str = ""
    max_iter: int | None = None
    time_budget: float | None = None
    refresh_data: bool = False
    refresh_source: bool = False
    auto_confirm: bool = False
    instructions: tuple[str, ...] = ()


def add_experiment_run_options(
    parser: argparse.ArgumentParser,
    *,
    include_project_flags: bool = False,
    include_confirmation: bool = False,
) -> None:
    if include_project_flags:
        parser.add_argument("--project", default="")
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--namespace", default="")
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--time-budget", type=float, default=None)
    if include_confirmation:
        parser.add_argument("--auto-confirm", action="store_true")
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--refresh-source", action="store_true")
    parser.add_argument("--instruction", "--constraint", action="append", default=[])


def options_from_namespace(args: argparse.Namespace) -> ExperimentRunOptions:
    return ExperimentRunOptions(
        project=str(getattr(args, "project", "") or ""),
        dry_run=bool(getattr(args, "dry_run", False)),
        namespace=str(getattr(args, "namespace", "") or ""),
        max_iter=getattr(args, "max_iter", None),
        time_budget=getattr(args, "time_budget", None),
        refresh_data=bool(getattr(args, "refresh_data", False)),
        refresh_source=bool(getattr(args, "refresh_source", False)),
        auto_confirm=bool(getattr(args, "auto_confirm", False)),
        instructions=tuple(
            str(item).strip()
            for item in getattr(args, "instruction", [])
            if str(item).strip()
        ),
    )


def skill_command_args(options: ExperimentRunOptions, *, project: str) -> list[str]:
    args = ["experiment", "run", "--project", project]
    if options.dry_run:
        args.append("--dry-run")
    if options.namespace:
        args.extend(["--namespace", options.namespace])
    if options.max_iter is not None:
        args.extend(["--max-iter", str(options.max_iter)])
    if options.time_budget is not None:
        args.extend(["--time-budget", str(options.time_budget)])
    if options.refresh_data:
        args.append("--refresh-data")
    if options.refresh_source:
        args.append("--refresh-source")
    if options.auto_confirm:
        args.append("--auto-confirm")
    for instruction in options.instructions:
        args.extend(["--instruction", instruction])
    return args


__all__ = [
    "ExperimentRunOptions",
    "add_experiment_run_options",
    "options_from_namespace",
    "skill_command_args",
]
