"""Experiment CLI action handlers."""

from __future__ import annotations

import argparse
import subprocess

from automl.agent.launch import build_launch
from automl.agent.proposer_context import gather_proposer_context
from automl.agent.run_options import options_from_namespace, skill_command_args
from automl.experiment.cleanup import delete as delete_experiment
from automl.experiment.views import compare, leaderboard
from automl.experiment.views.summary import build_summary
from automl.experiment.views.summary import experiments as list_experiments

from ._common import print_json, session_from_args


def _session(args: argparse.Namespace):
    return session_from_args(args, experiment_id=getattr(args, "experiment_id_arg", None))


def _list(args: argparse.Namespace) -> int:
    print_json(list_experiments(session=session_from_args(args)))
    return 0


def _run(args: argparse.Namespace) -> int:
    active = _session(args)
    options = options_from_namespace(args)
    automl_args = skill_command_args(options, project=active.project_name)
    launch = build_launch(
        session=active,
        automl_args=automl_args,
        max_budget_usd=args.max_budget_usd,
        output_format=args.output_format,
        claude_bin=args.claude_bin,
    )
    completed = subprocess.run(
        launch.command,
        env=launch.env,
        cwd=launch.cwd,
        check=False,
    )
    if args.json:
        print_json(
            {
                "returncode": completed.returncode,
                "command": launch.command,
                "cwd": launch.cwd,
            }
        )
    return int(completed.returncode)


def _delete(args: argparse.Namespace) -> int:
    active = _session(args)
    print_json(
        delete_experiment(
            args.experiment_id_arg,
            apply=args.apply,
            hard_delete=args.hard_delete,
            backend_store_uri=args.backend_store_uri,
            artifacts_destination=args.artifacts_destination,
            session=active,
        )
    )
    return 0


def _leaderboard(args: argparse.Namespace) -> int:
    print_json(
        leaderboard(
            metric=args.metric,
            n=args.n,
            training_origin=args.training_origin,
            session=_session(args),
        )
    )
    return 0


def _compare(args: argparse.Namespace) -> int:
    print_json(compare(args.run_ids, session=session_from_args(args)))
    return 0


def _summary(args: argparse.Namespace) -> int:
    print_json(build_summary(session=_session(args)))
    return 0


def _proposer_context(args: argparse.Namespace) -> int:
    print_json(
        gather_proposer_context(
            metric=args.metric,
            n_top=args.n_top,
            session=_session(args),
        )
    )
    return 0
