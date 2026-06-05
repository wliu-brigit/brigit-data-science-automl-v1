"""Project CLI verbs."""

from __future__ import annotations

import argparse
from pathlib import Path

from automl.project import allowed_dependencies, create_project, list_projects
from automl.project.cleanup import delete as delete_project

from ._common import print_json, session_from_args


def _add_cleanup_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--scope", choices=["project", "qa"], default="project")


def _cleanup_kwargs(args: argparse.Namespace) -> dict:
    return {
        "apply": args.apply,
        "scope": args.scope,
    }


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("project")
    project_sub = parser.add_subparsers(dest="action", required=True)

    project_sub.add_parser("list").set_defaults(func=_list)
    project_sub.add_parser("deps").set_defaults(func=_deps)

    init = project_sub.add_parser("init")
    init.add_argument("name")
    init.add_argument("--template", default="snowflake", choices=["snowflake"])
    init.set_defaults(func=_init)

    delete = project_sub.add_parser("delete")
    delete.add_argument("name", nargs="?")
    _add_cleanup_flags(delete)
    delete.set_defaults(func=_delete)


def _list(args: argparse.Namespace) -> int:
    print_json(list_projects(repo_root=args.project_root))
    return 0


def _deps(args: argparse.Namespace) -> int:
    print_json(allowed_dependencies(session=session_from_args(args)))
    return 0


def _init(args: argparse.Namespace) -> int:
    root = args.project_root or Path.cwd()
    print_json(create_project(args.name, project_root=root, template=args.template))
    return 0


def _delete(args: argparse.Namespace) -> int:
    active = session_from_args(args)
    print_json(delete_project(args.name, session=active, **_cleanup_kwargs(args)))
    return 0


__all__ = ["add_parser"]
