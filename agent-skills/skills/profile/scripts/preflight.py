from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path


def parse_arguments(raw: str) -> dict:
    try:
        tokens = shlex.split(raw)
    except ValueError as exc:
        return {"dry_run": False, "mode": "error", "error": str(exc)}

    return {
        "dry_run": "--dry-run" in tokens,
        "mode": "profile",
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

    payload = parse_arguments(args.arguments)
    payload["project_root"] = str(args.project_root)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
