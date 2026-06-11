"""Validate CLI verbs."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import _validate_actions as actions


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("validate")
    validate_sub = parser.add_subparsers(dest="target", required=True)

    project = validate_sub.add_parser("project")
    project.add_argument(
        "--probe-snowflake",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="force the Snowflake live probe on/off, overriding RUN_CONFIG.skip_snowflake_live_check",
    )
    project.set_defaults(func=actions._project)

    model = validate_sub.add_parser("model")
    model.add_argument("--module", required=True)
    model.add_argument("--class-name", required=True)
    model.set_defaults(func=actions._model)

    proposal = validate_sub.add_parser("proposal")
    proposal.add_argument("--proposal-json", required=True, help="Path to proposal JSON, or '-' for stdin")
    proposal.add_argument("--output", type=Path)
    proposal.set_defaults(func=actions._proposal)


__all__ = ["add_parser"]
