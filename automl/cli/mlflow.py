"""MLflow administrative CLI verbs."""

from __future__ import annotations

import argparse

from automl.mlflow.cleanup import purge as purge_mlflow

from ._common import print_json, session_from_args


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("mlflow")
    mlflow_sub = parser.add_subparsers(dest="action", required=True)

    purge = mlflow_sub.add_parser("purge")
    purge.add_argument("name", nargs="?")
    purge.add_argument("--scope", choices=["qa", "deleted"])
    purge.add_argument("--apply", action="store_true")
    purge.add_argument("--backend-store-uri", default="")
    purge.add_argument("--artifacts-destination", default="")
    purge.set_defaults(func=_purge)


def _purge(args: argparse.Namespace) -> int:
    active = session_from_args(args)
    print_json(
        purge_mlflow(
            args.name,
            scope=args.scope,
            apply=args.apply,
            backend_store_uri=args.backend_store_uri,
            artifacts_destination=args.artifacts_destination,
            session=active,
        )
    )
    return 0


__all__ = ["add_parser"]
