"""Trial CLI verbs."""

from __future__ import annotations

from pathlib import Path

from . import _trial_actions as actions


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("trial")
    trial_sub = parser.add_subparsers(dest="action", required=True)

    list_parser = trial_sub.add_parser("list")
    list_parser.add_argument("experiment_id_arg", nargs="?")
    list_parser.set_defaults(func=actions._list)

    create = trial_sub.add_parser("create")
    create.add_argument("slug", nargs="?")
    create.add_argument("--strategy")
    create.add_argument("--hypothesis", default="")
    create.add_argument("--seed")
    create.add_argument("--model-source", type=Path)
    create.add_argument("--training-origin", default="automl", choices=["automl", "human"])
    create.add_argument("--proposal-json", type=Path)
    create.set_defaults(func=actions._create)

    fork = trial_sub.add_parser("fork")
    fork.add_argument("slug")
    fork.add_argument("--seed", default="best")
    fork.add_argument("--strategy", default="manual_fork")
    fork.add_argument("--hypothesis", default="")
    fork.set_defaults(func=actions._fork)

    promote = trial_sub.add_parser("promote")
    promote.add_argument("slug")
    promote.add_argument("--model-path", required=True, type=Path)
    promote.add_argument("--hypothesis", required=True)
    promote.add_argument("--strategy", default="manual_promote")
    promote.set_defaults(func=actions._promote)

    run = trial_sub.add_parser("run")
    run.add_argument("path")
    run.add_argument("--dataset-id")
    run.set_defaults(func=actions._run)

    show = trial_sub.add_parser("show")
    show.add_argument("run_id")
    show.set_defaults(func=actions._show)

    delete = trial_sub.add_parser("delete")
    delete.add_argument("run_id")
    delete.add_argument("--apply", action="store_true")
    delete.set_defaults(func=actions._delete)

    lock = trial_sub.add_parser("lock")
    lock_sub = lock.add_subparsers(dest="lock_action", required=True)
    acquire = lock_sub.add_parser("acquire")
    acquire.add_argument("--session-id", default="")
    acquire.set_defaults(func=actions._lock_acquire)
    release = lock_sub.add_parser("release")
    release.add_argument("--session-id", default="")
    release.add_argument("--lock-id", default="")
    release.set_defaults(func=actions._lock_release)


__all__ = ["add_parser"]
