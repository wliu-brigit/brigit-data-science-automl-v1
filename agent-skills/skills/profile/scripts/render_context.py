from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def _load_preflight(skill_scripts_dir: Path):
    path = skill_scripts_dir / "preflight.py"
    spec = importlib.util.spec_from_file_location("profile_preflight", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load preflight module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_context(project_root: Path, arguments: str) -> dict:
    preflight = _load_preflight(Path(__file__).resolve().parent)
    return {
        "schema_version": 1,
        "operation": "profile",
        "writes_during_render": False,
        "invocation": preflight.parse_arguments(arguments),
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
