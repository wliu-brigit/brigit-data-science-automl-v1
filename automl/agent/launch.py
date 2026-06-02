"""Claude Code launch builder for the agent loop."""

from __future__ import annotations

import json
import os
import shlex
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from automl.project import Session, session as active_project_session


@dataclass(frozen=True)
class LaunchSpec:
    command: list[str]
    env: dict[str, str]
    cwd: Path


@dataclass(frozen=True)
class ClaudeRole:
    model: str = ""
    effort: str = ""


def build_launch(
    *,
    session: Session | None = None,
    automl_args: list[str] | None = None,
    max_budget_usd: str = "5",
    output_format: str = "text",
    claude_bin: str = "claude",
    permission_mode: str = "bypassPermissions",
) -> LaunchSpec:
    """Build the single subprocess invocation that drives the LLM loop."""

    active = session if session is not None else active_project_session()
    project_root = active.config.repo_root
    plugin_dir = project_root / "agent-skills"
    project = active.project_name
    models = _model_settings(active)
    manager = models["manager"]
    normalized_args = _normalize_automl_args(project=project, automl_args=automl_args or [])

    env = os.environ.copy()
    session_id = str(env.get("AUTOML_SESSION_ID") or env.get("CLAUDE_SESSION_ID") or uuid.uuid4())
    env["AUTOML_SESSION_ID"] = session_id
    env["CLAUDE_SESSION_ID"] = session_id
    env["AUTOML_PROJECT_ROOT"] = str(project_root)
    env["AUTOML_PROJECT"] = project
    env["AUTOML_EXPERIMENT_ID"] = active.active_experiment_id
    env["AUTOML_NAMESPACE"] = active.namespace
    env["AUTOML_INHERIT_DRY_RUN"] = "1" if active.dry_run else "0"

    command = [claude_bin, "--session-id", session_id]
    if manager.model:
        command.extend(["--model", manager.model])
    if manager.effort:
        command.extend(["--effort", manager.effort])
    command.extend(
        [
            "--agents",
            json.dumps(
                _agent_overrides(plugin_dir, models),
                separators=(",", ":"),
            ),
            "--strict-mcp-config",
            "--add-dir",
            str(project_root),
            "--plugin-dir",
            str(plugin_dir),
            "--print",
            "--permission-mode",
            permission_mode,
            "--output-format",
            output_format,
        ]
    )
    if max_budget_usd:
        command.extend(["--max-budget-usd", max_budget_usd])
    if output_format == "stream-json":
        command.append("--verbose")
    command.append("/brigit-automl:automl " + shlex.join(normalized_args))
    return LaunchSpec(command=command, env=env, cwd=project_root)


def _model_settings(active: Session) -> dict[str, ClaudeRole]:
    models = active.config.models
    return {
        "manager": _role_settings(getattr(models, "manager", None)),
        "proposer": _role_settings(getattr(models, "proposer", None)),
        "coder": _role_settings(getattr(models, "coder", None)),
    }


def _role_settings(raw: Any) -> ClaudeRole:
    return ClaudeRole(
        model=str(getattr(raw, "model", "") or "").strip(),
        effort=str(getattr(raw, "effort", "") or "").strip(),
    )


def _parse_agent_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"agent file {path} missing YAML frontmatter")
    _, frontmatter, prompt = text.split("---", 2)
    loaded = yaml.safe_load(frontmatter) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"agent file {path} frontmatter must be a mapping")
    tools = loaded.get("tools")
    if isinstance(tools, str):
        loaded["tools"] = [item.strip() for item in tools.split(",") if item.strip()]
    loaded["prompt"] = prompt.strip()
    return loaded


def _agent_overrides(plugin_dir: Path, role_settings: dict[str, ClaudeRole]) -> dict[str, Any]:
    mapping = {
        "automl-proposer": role_settings["proposer"],
        "automl-coder": role_settings["coder"],
    }
    overrides: dict[str, Any] = {}
    for agent_name, settings in mapping.items():
        agent = _parse_agent_file(plugin_dir / "agents" / f"{agent_name}.md")
        if settings.model:
            agent["model"] = settings.model
        if settings.effort:
            agent["effort"] = settings.effort
        overrides[agent_name] = agent
    return overrides


def _normalize_automl_args(*, project: str, automl_args: list[str]) -> list[str]:
    args = list(automl_args)
    if not args:
        return ["experiment", "run", "--project", project]
    if args[:2] != ["experiment", "run"]:
        return args

    project_value = ""
    index = 2
    while index < len(args):
        item = args[index]
        if item == "--project":
            if index + 1 >= len(args):
                raise ValueError("inner automl experiment run --project is missing a value")
            project_value = args[index + 1]
            break
        if item.startswith("--project="):
            project_value = item.split("=", 1)[1]
            break
        index += 1

    if project_value:
        if project_value != project:
            raise ValueError(
                f"inner automl experiment run --project {project_value!r} does not match "
                f"launcher --project {project!r}"
            )
        return args
    return ["experiment", "run", "--project", project, *args[2:]]


__all__ = ["ClaudeRole", "LaunchSpec", "build_launch"]
