"""Experiment CLI verbs."""

from __future__ import annotations

from automl.agent.run_options import add_experiment_run_options

from . import _experiment_actions as actions


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("experiment")
    experiment_sub = parser.add_subparsers(dest="action", required=True)

    list_parser = experiment_sub.add_parser("list")
    list_parser.set_defaults(func=actions._list)

    run = experiment_sub.add_parser("run")
    run.add_argument("experiment_id_arg", nargs="?")
    run.add_argument("--max-budget-usd", default="5")
    run.add_argument("--output-format", choices=["text", "json", "stream-json"], default="text")
    run.add_argument("--claude-bin", default="claude")
    run.add_argument("--json", action="store_true")
    add_experiment_run_options(run, include_confirmation=True)
    run.set_defaults(func=actions._run)

    delete = experiment_sub.add_parser("delete")
    delete.add_argument("experiment_id_arg")
    delete.add_argument("--apply", action="store_true")
    delete.set_defaults(func=actions._delete)

    board = experiment_sub.add_parser("leaderboard")
    board.add_argument("experiment_id_arg", nargs="?")
    board.add_argument("--metric")
    board.add_argument("--n", type=int, default=10)
    board.add_argument("--training-origin")
    board.set_defaults(func=actions._leaderboard)

    comp = experiment_sub.add_parser("compare")
    comp.add_argument("run_ids", nargs="+")
    comp.set_defaults(func=actions._compare)

    summary = experiment_sub.add_parser("summary")
    summary.add_argument("experiment_id_arg", nargs="?")
    summary.set_defaults(func=actions._summary)

    context = experiment_sub.add_parser("proposer-context")
    context.add_argument("experiment_id_arg", nargs="?")
    context.add_argument("--metric")
    context.add_argument("--n-top", type=int, default=10)
    context.set_defaults(func=actions._proposer_context)


__all__ = ["add_parser"]
