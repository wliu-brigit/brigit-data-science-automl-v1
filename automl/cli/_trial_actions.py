"""Trial CLI action handlers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from automl.mlflow.experiment import list_trials
from automl.runner import run_trial
from automl.runner import session_lock as trial_lock
from automl.runner.results import trial_result_exit_code
from automl.trial.cleanup import delete as delete_trial
from automl.trial.create import create as create_trial
from automl.trial.fork import fork as fork_trial
from automl.trial.show import show_trial

from ._common import print_json, session_from_args


def _list(args: argparse.Namespace) -> int:
    active = session_from_args(args, experiment_id=getattr(args, "experiment_id_arg", None))
    print_json(list_trials(experiment_id=active.active_experiment_id))
    return 0


def _create(args: argparse.Namespace) -> int:
    active = session_from_args(args)
    proposal = _load_proposal(args.proposal_json) if args.proposal_json else None
    trial_path = create_trial(
        slug=args.slug,
        strategy=args.strategy,
        hypothesis=args.hypothesis,
        seed=args.seed,
        model_source=args.model_source,
        training_origin=args.training_origin,
        proposal=proposal,
        session=active,
    )
    print_json({"trial_dir": trial_path})
    return 0


def _fork(args: argparse.Namespace) -> int:
    active = session_from_args(args)
    trial_path = fork_trial(
        slug=args.slug,
        seed=args.seed,
        strategy=args.strategy,
        hypothesis=args.hypothesis,
        session=active,
    )
    print_json({"trial_dir": trial_path})
    return 0


def _promote(args: argparse.Namespace) -> int:
    active = session_from_args(args)
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"model_path does not exist: {model_path}")
    trial_path = create_trial(
        slug=args.slug,
        strategy=args.strategy,
        hypothesis=args.hypothesis,
        model_source=model_path,
        training_origin="human",
        session=active,
    )
    result = run_trial(trial_path, session=active)
    print_json(result)
    return trial_result_exit_code(result)


def _run(args: argparse.Namespace) -> int:
    active = session_from_args(args)
    result = run_trial(args.path, session=active, dataset_id=getattr(args, "dataset_id", None))
    print_json(result)
    return trial_result_exit_code(result)


def _load_proposal(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("proposal JSON must be an object")
    return loaded


def _show(args: argparse.Namespace) -> int:
    print_json(show_trial(args.run_id, session=session_from_args(args)))
    return 0


def _delete(args: argparse.Namespace) -> int:
    active = session_from_args(args)
    print_json(
        delete_trial(
            args.run_id,
            apply=args.apply,
            session=active,
        )
    )
    return 0


def _lock_acquire(args: argparse.Namespace) -> int:
    session_id = args.session_id or "human-cli"
    active = session_from_args(args)
    print_json(trial_lock.acquire_for_session(active, session_id=session_id))
    return 0


def _lock_release(args: argparse.Namespace) -> int:
    active = session_from_args(args)
    print_json(
        trial_lock.release_for_session(
            active,
            session_id=args.session_id,
            lock_id=args.lock_id,
        )
    )
    return 0
