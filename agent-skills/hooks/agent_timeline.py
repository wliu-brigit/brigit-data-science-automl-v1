#!/usr/bin/env python3
"""Thin Claude hook transport for agent timeline events."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("AUTOML_PROJECT_ROOT") or Path.cwd()).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=_default_project_root())
    parser.add_argument("--project", default=os.environ.get("AUTOML_PROJECT"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    hook_event = subparsers.add_parser("hook-event")
    hook_event.set_defaults(func=_hook_event)

    publish_cmd = subparsers.add_parser("publish")
    publish_cmd.add_argument("--session-id")
    publish_cmd.set_defaults(func=_publish)

    args = parser.parse_args(argv)
    session = _bootstrap_session(args.project_root, args.project)
    return int(args.func(args, session))


def _hook_event(args: argparse.Namespace, session) -> int:
    del args
    from automl.agent.timeline import handle_event

    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError as exc:
        print(f"invalid hook json: {exc}", file=sys.stderr)
        return 2
    if not isinstance(payload, dict):
        print("hook input must be a JSON object", file=sys.stderr)
        return 2
    print(json.dumps(handle_event(payload, session=session), indent=2, default=str))
    return 0


def _publish(args: argparse.Namespace, session) -> int:
    from automl.agent.timeline import publish

    print(
        json.dumps(
            publish(session_id=args.session_id, session=session),
            indent=2,
            default=str,
        )
    )
    return 0


def _bootstrap_session(project_root: Path, project: str | None):
    from automl.project import use_project

    root = project_root.resolve()
    return use_project(
        project or _single_project_name(root),
        repo_root=root,
        dry_run=os.environ.get("AUTOML_INHERIT_DRY_RUN", "") == "1",
        namespace=os.environ.get("AUTOML_NAMESPACE", ""),
        experiment_id=os.environ.get("AUTOML_EXPERIMENT_ID") or None,
    )


def _single_project_name(project_root: Path) -> str:
    projects_dir = project_root / "projects"
    candidates = sorted(
        item.name
        for item in projects_dir.iterdir()
        if item.is_dir() and not item.name.startswith("__")
    )
    if len(candidates) != 1:
        raise SystemExit(
            "could not infer project; pass --project or set AUTOML_PROJECT",
        )
    return candidates[0]


def _default_project_root() -> Path:
    return PROJECT_ROOT


if __name__ == "__main__":
    raise SystemExit(main())
