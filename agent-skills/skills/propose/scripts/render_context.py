from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path


def _invocation(arguments: str) -> dict:
    try:
        tokens = shlex.split(arguments)
    except ValueError as exc:
        return {"dry_run": False, "mode": "error", "error": str(exc)}

    return {
        "dry_run": "--dry-run" in tokens,
        "mode": "propose",
    }


def build_context(project_root: Path, arguments: str) -> dict:
    return {
        "schema_version": 1,
        "operation": "propose",
        "writes_during_render": False,
        "invocation": _invocation(arguments),
        "project": {"root": str(project_root)},
    }


def _parse_cli(argv: list[str]) -> argparse.Namespace:
    arguments = ""
    if "--arguments" in argv:
        index = argv.index("--arguments")
        if index + 1 < len(argv):
            arguments = argv[index + 1]
            del argv[index : index + 2]
        else:
            del argv[index]

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    args = parser.parse_args(argv)
    args.arguments = arguments
    return args


def main() -> int:
    args = _parse_cli(sys.argv[1:])

    print(json.dumps(build_context(args.project_root, args.arguments), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
