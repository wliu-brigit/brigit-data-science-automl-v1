"""Validate CLI action handlers."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

from automl.agent import Proposal, validate_proposal
from automl.data.synthetic import make_synthetic_fixture
from automl.errors import ProjectError
from automl.model import validate_model
from automl.project import validate_project

from ._common import print_json, session_from_args


def _project(args: argparse.Namespace) -> int:
    report = validate_project(
        session=session_from_args(args),
        live=True,
        probe_snowflake=getattr(args, "probe_snowflake", None),
    )
    print_json(report)
    return 0 if report.passed else 1


def _model(args: argparse.Namespace) -> int:
    module = importlib.import_module(args.module)
    cls = getattr(module, args.class_name)
    df, registry = make_synthetic_fixture()
    active = _optional_session(args)
    report = validate_model(cls, df=df, registry=registry, session=active)
    print_json(report)
    return 0 if report.passed else 1


def _proposal(args: argparse.Namespace) -> int:
    active = _optional_session(args)
    payload = _read_json_arg(args.proposal_json)
    report = validate_proposal(proposal=payload, session=active)
    print_json(report)
    if report.passed and args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(Proposal.from_dict(payload).to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
    return 0 if report.passed else 1


def _optional_session(args: argparse.Namespace):
    try:
        return session_from_args(args)
    except ProjectError:
        if (
            args.project
            or args.project_root
            or args.dry_run
            or args.namespace
            or args.experiment_id
        ):
            raise
        return None


def _read_json_arg(value: str) -> dict[str, Any]:
    raw = sys.stdin.read() if value == "-" else Path(value).read_text(encoding="utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise SystemExit("proposal JSON must be an object")
    return payload
