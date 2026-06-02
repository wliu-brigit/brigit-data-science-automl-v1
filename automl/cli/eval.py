"""Eval CLI verbs."""

from __future__ import annotations

import argparse

from automl.eval import evaluate, list_eval_datasets

from ._common import print_json, session_from_args


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("eval")
    eval_sub = parser.add_subparsers(dest="action", required=True)

    list_parser = eval_sub.add_parser("list")
    list_parser.set_defaults(func=_list)

    compute = eval_sub.add_parser("compute")
    compute.add_argument("--model-run-id", required=True)
    compute.add_argument("--eval-dataset", required=True)
    compute.add_argument("--label", required=True)
    compute.add_argument("--overwrite", action="store_true")
    compute.add_argument("--set-as-primary-label", action="store_true")
    compute.set_defaults(func=_compute)


def _list(args: argparse.Namespace) -> int:
    print_json(list_eval_datasets(session=session_from_args(args)))
    return 0


def _compute(args: argparse.Namespace) -> int:
    print_json(
        evaluate(
            session=session_from_args(args),
            model_run_id=args.model_run_id,
            eval_dataset_id=args.eval_dataset,
            label=args.label,
            overwrite=args.overwrite,
            set_as_primary_label=args.set_as_primary_label,
        )
    )
    return 0


__all__ = ["add_parser"]
