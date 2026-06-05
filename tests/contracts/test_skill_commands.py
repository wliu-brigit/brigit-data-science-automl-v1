from __future__ import annotations

import importlib.util
import re
import shlex
from pathlib import Path

import pytest

from automl.cli import build_parser

pytestmark = pytest.mark.contract

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_COMMAND_ROOTS = [
    REPO_ROOT / "agent-skills" / "skills",
    REPO_ROOT / "agent-skills" / "agents",
    REPO_ROOT / "agent-skills" / "references",
]
ACTIVE_PROJECT_DOC_ROOTS = [
    REPO_ROOT / "projects" / "example_homecredit",
]

RETIRED_EXECUTABLE_PATTERNS = {
    "top-level automl run": re.compile(r"(?<!experiment )(?<!:automl )\bautoml run\b"),
    "slash automl run": re.compile(r"/brigit-automl:automl run\b"),
    "automl inspect": re.compile(r"\bautoml inspect\b"),
    "automl loop-context": re.compile(r"\bautoml loop-context\b"),
    "automl profile": re.compile(r"\bautoml profile\b"),
    "automl propose validate": re.compile(r"\bautoml propose validate\b"),
    "automl project create": re.compile(r"\bautoml project create\b"),
    "python module dependencies": re.compile(r"\bpython -m automl\.core\.dependencies\b"),
    "python module lock": re.compile(r"\bpython -m automl\.session\.lock\b"),
    "legacy core import": re.compile(r"\bfrom automl\.core\.|\bimport automl\.core\."),
    "hook route flag": re.compile(r"\s--route(?:\s|=)"),
    "hook publish flag": re.compile(r"\s--publish-mlflow\b"),
    # Retired data-layer vocabulary (snowflake-source-and-split-keys effort,
    # steps 1-4): renamed or hard-cut; must not resurface in skills or docs.
    "retired hash_key field": re.compile(r"\bhash_key\b"),
    "retired SPLITID column": re.compile(r"\bSPLITID\b"),
    "retired base_data_sql field": re.compile(r"\bbase_data_sql\b"),
    "retired split_range kwarg": re.compile(r"\bsplit_range\b"),
    "retired splits bucket API": re.compile(r"\btrain_buckets\b|\btest_buckets\b|\.buckets\("),
    "retired bucket-range Splits": re.compile(r"Splits\(train=\[\("),
    "retired route_namespace name": re.compile(r"\broute_namespace\b"),
    "retired dry-run transport env": re.compile(r"\bAUTOML_DRY_RUN\b"),
    "retired data prepare api": re.compile(r"\bprepare_data\("),
    "retired data snapshot loader": re.compile(r"\bload_data_snapshot\("),
    "retired data tuple loader": re.compile(r"\bload_data\("),
    "retired prepare_snapshot safe command": re.compile(r"\bprepare_snapshot\b"),
    "retired pipeline prepare method": re.compile(r"\bDataPipeline\.prepare_data\b"),
    "retired active snapshot wording": re.compile(r"\bactive snapshot\b", re.IGNORECASE),
    "retired data snapshot wording": re.compile(r"\bdata snapshot\b", re.IGNORECASE),
    "retired snapshot registry wording": re.compile(r"\bsnapshot registry\b", re.IGNORECASE),
    "retired snapshot_name field": re.compile(r"\bsnapshot_name\b"),
    "retired active_data_snapshot field": re.compile(r"\bactive_data_snapshot\b"),
    "retired snapshot_usage field": re.compile(r"\bsnapshot_usage\b"),
    "retired eval snapshot loader": re.compile(r"\bload_eval_snapshot\b"),
    "retired eval snapshot preparer": re.compile(r"\bprepare_eval_snapshot\b"),
    "retired eval snapshot module": re.compile(r"\bautoml\.eval\.snapshot\b"),
    "retired eval snapshot field": re.compile(r"\beval_snapshot_id\b"),
    "retired auto-confirm env": re.compile(r"\bAUTOML_AUTO_CONFIRM\b"),
    "retired dataset notebook name": re.compile(
        r"\b(?:1_define_data_and_snapshot|2_profile_data_snapshot)\.ipynb\b"
    ),
    "late project flag": re.compile(
        r"\buv run automl (?:project|experiment|trial|data|eval|validate)\b[^\n`]*\s--project\b"
    ),
    "late project-root flag": re.compile(
        r"\buv run automl (?:project|experiment|trial|data|eval|validate)\b[^\n`]*\s--project-root\b"
    ),
    "late dry-run flag": re.compile(
        r"\buv run automl (?:project|experiment|trial|data|eval|validate)\b[^\n`]*\s--dry-run\b"
    ),
}


def _skill_files() -> list[Path]:
    files: list[Path] = []
    for root in SKILL_COMMAND_ROOTS:
        files.extend(path for path in root.rglob("*") if path.suffix in {".md", ".py"})
    return sorted(files)


def _active_project_doc_files() -> list[Path]:
    files: list[Path] = []
    for root in ACTIVE_PROJECT_DOC_ROOTS:
        if root.is_file():
            files.append(root)
        else:
            files.extend(path for path in root.rglob("*") if path.suffix in {".md", ".ipynb"})
    return sorted(files)


def _automl_argv(command: str) -> list[str] | None:
    tokens = shlex.split(command)
    while tokens and "=" in tokens[0] and not tokens[0].startswith("-"):
        tokens = tokens[1:]
    if tokens[:2] == ["uv", "run"]:
        tokens = tokens[2:]
    if not tokens or tokens[0] != "automl":
        return None
    return tokens[1:]


def _assert_parses(command: str) -> None:
    argv = _automl_argv(command)
    if argv is None:
        return
    build_parser().parse_args(argv)


def test_cli_json_flag_surface_is_contract_pinned():
    parser = build_parser()

    def subparser(*path: str):
        current = parser
        for name in path:
            choices = {}
            for action in current._actions:
                choices.update(getattr(action, "choices", {}) or {})
            current = choices[name]
        return current

    def options(parser):
        return {
            item
            for action in parser._actions
            for item in getattr(action, "option_strings", ())
        }

    assert "--json" in options(subparser("experiment", "run"))
    for command in [
        ("project", "list"),
        ("experiment", "list"),
        ("trial", "run"),
        ("data", "materialize"),
        ("eval", "compute"),
        ("validate", "proposal"),
    ]:
        assert "--json" not in options(subparser(*command)), command


def test_skill_facing_files_do_not_reference_retired_executable_commands():
    offenders = []
    for path in _skill_files():
        text = path.read_text(encoding="utf-8")
        for label, pattern in RETIRED_EXECUTABLE_PATTERNS.items():
            if pattern.search(text):
                offenders.append((path.relative_to(REPO_ROOT).as_posix(), label))

    assert offenders == []


def test_active_project_docs_do_not_reference_retired_cutover_terms():
    offenders = []
    for path in _active_project_doc_files():
        text = path.read_text(encoding="utf-8")
        for label, pattern in RETIRED_EXECUTABLE_PATTERNS.items():
            if pattern.search(text):
                offenders.append((path.relative_to(REPO_ROOT).as_posix(), label))

    assert offenders == []


def test_automl_render_context_safe_commands_use_current_cli_surface():
    module_path = REPO_ROOT / "agent-skills" / "skills" / "automl" / "scripts" / "render_context.py"
    spec = importlib.util.spec_from_file_location("automl_skill_render_context", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    context = module.build_context(
        REPO_ROOT,
        "experiment run --project example_homecredit --dry-run --max-iter 1",
    )
    safe_commands = context["safe_commands"]
    project_contract = context["project_contract"]

    for command in safe_commands.values():
        _assert_parses(command)

    assert "automl --project example_homecredit" in safe_commands["loop_context"]
    assert " experiment proposer-context " in f" {safe_commands['loop_context']} "
    assert " data materialize" in safe_commands["materialize_dataset"]
    assert " validate proposal" in safe_commands["persist_proposal"]
    assert " validate proposal" in safe_commands["validate_proposal"]
    assert " --json" not in safe_commands["loop_context"]
    assert " --json" not in safe_commands["materialize_dataset"]
    assert " --json" not in safe_commands["persist_proposal"]
    assert " --json" not in safe_commands["validate_proposal"]
    assert " --proposal-json -" in safe_commands["persist_proposal"]
    assert " --proposal-json '<proposal.json>'" in safe_commands["validate_proposal"]
    assert "automl propose validate" not in safe_commands["persist_proposal"]
    assert "automl loop-context" not in safe_commands["loop_context"]
    assert "--route" not in safe_commands["timeline_publish"]
    assert "--publish-mlflow" not in safe_commands["timeline_publish"]
    assert project_contract["target_column"] == "target"
    assert project_contract["raw_target_column"] == "TARGET"
    assert project_contract["primary_metric"] == "auc"
    assert project_contract["required_transformers"][0]["name"] == "homecredit_organization_woe"
    assert project_contract["required_transformers"][0]["columns"] == ["organization_type"]


def _load_render_context_module():
    module_path = REPO_ROOT / "agent-skills" / "skills" / "automl" / "scripts" / "render_context.py"
    spec = importlib.util.spec_from_file_location("automl_skill_render_context", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_automl_render_context_forwards_refresh_source_to_materialize():
    module = _load_render_context_module()

    context = module.build_context(
        REPO_ROOT,
        "experiment run --project example_homecredit --dry-run --refresh-source",
    )

    materialize = context["safe_commands"]["materialize_dataset"]
    _assert_parses(materialize)
    assert materialize.endswith(" data materialize --refresh-source")


def test_automl_render_context_forwards_refresh_data_to_materialize():
    module = _load_render_context_module()

    context = module.build_context(
        REPO_ROOT,
        "experiment run --project example_homecredit --dry-run --refresh-data",
    )

    materialize = context["safe_commands"]["materialize_dataset"]
    _assert_parses(materialize)
    assert materialize.endswith(" data materialize --refresh-data")


def test_automl_render_context_resolves_repo_root_from_inside_project():
    module = _load_render_context_module()

    context = module.build_context(
        REPO_ROOT / "projects" / "example_homecredit",
        "experiment run --project example_homecredit --dry-run",
    )

    assert context["invocation"]["mode"] == "run"
    assert context["project"]["root"] == str(REPO_ROOT)
    assert context["route"] == "dry_run/example_homecredit/example-homecredit"
