"""Data CLI verbs."""

from __future__ import annotations

import argparse

from automl.data import activate_dataset, list_datasets, materialize, profile

from ._common import print_json, session_from_args


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("data")
    data_sub = parser.add_subparsers(dest="action", required=True)

    list_parser = data_sub.add_parser("list")
    list_parser.set_defaults(func=_list)

    profile_parser = data_sub.add_parser("profile")
    profile_parser.add_argument("dataset_id", nargs="?")
    profile_parser.set_defaults(func=_profile)

    activate_parser = data_sub.add_parser("activate")
    activate_parser.add_argument("dataset_id")
    activate_parser.set_defaults(func=_activate)

    materialize_parser = data_sub.add_parser("materialize")
    materialize_parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="re-derive the dataset from the source (default attaches to the pinned active dataset)",
    )
    materialize_parser.add_argument(
        "--refresh-source",
        action="store_true",
        help="rebuild the source's upstream (Snowflake base table) first; implies --refresh-data",
    )
    materialize_parser.set_defaults(func=_materialize)


def _list(args: argparse.Namespace) -> int:
    print_json(list_datasets(session=session_from_args(args)))
    return 0


def _profile(args: argparse.Namespace) -> int:
    print_json(profile(dataset_id=args.dataset_id, session=session_from_args(args)))
    return 0


def _activate(args: argparse.Namespace) -> int:
    print_json(activate_dataset(args.dataset_id, session=session_from_args(args)))
    return 0


def _materialize(args: argparse.Namespace) -> int:
    dataset = materialize(
        refresh_data=args.refresh_data,
        refresh_source=args.refresh_source,
        include_rows=False,
        session=session_from_args(args),
    )
    print_json(dataset)
    return 0


__all__ = ["add_parser"]
