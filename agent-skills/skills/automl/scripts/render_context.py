from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shlex
import sys
import tomllib
from pathlib import Path
from typing import Any


def _load_preflight(skill_scripts_dir: Path):
    path = skill_scripts_dir / "preflight.py"
    spec = importlib.util.spec_from_file_location("automl_preflight", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load preflight module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_project_context(project_root: Path, project_name: str = "") -> Any | None:
    try:
        from automl.project import ProjectConfig, infer_project_name

        resolved_name = project_name or infer_project_name(repo_root=project_root, start=Path.cwd())
        return ProjectConfig.load(resolved_name, repo_root=project_root)
    except Exception:
        return None


def _load_project_context_error(project_root: Path, project_name: str = "") -> str:
    try:
        from automl.project import ProjectConfig, infer_project_name

        resolved_name = project_name or infer_project_name(repo_root=project_root, start=Path.cwd())
        ProjectConfig.load(resolved_name, repo_root=project_root)
    except Exception as exc:
        return str(exc)
    return ""


def _configured_project_names(project_root: Path) -> list[str]:
    projects_root = project_root / "projects"
    if not projects_root.exists():
        return []
    names: set[str] = set()
    for path in projects_root.glob("*/config.py"):
        if path.parent.is_dir():
            names.add(path.parent.name)
    return sorted(names)


def _fallback_project_name(project_root: Path, requested_project: str) -> str:
    if requested_project:
        return requested_project
    names = _configured_project_names(project_root)
    return names[0] if len(names) == 1 else ""


def _load_env_values(project_root: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    env_path = project_root / ".env"
    if not env_path.exists():
        return values
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def _env_value(project_root: Path, key: str) -> str:
    return os.environ.get(key, "") or _load_env_values(project_root).get(key, "")


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _role_payload(models: dict[str, Any], role: str) -> dict[str, str]:
    raw = models.get(role) if isinstance(models, dict) else getattr(models, role, None)
    if isinstance(raw, dict):
        return {
            "model": _text(raw.get("model")).strip(),
            "effort": _text(raw.get("effort")).strip(),
        }
    if raw is not None:
        return {
            "model": _text(getattr(raw, "model", "")).strip(),
            "effort": _text(getattr(raw, "effort", "")).strip(),
        }
    return {"model": "", "effort": ""}


def _models_payload(cfg: dict[str, Any]) -> dict[str, Any]:
    models = cfg.get("models") if isinstance(cfg, dict) else {}
    manager = _role_payload(models, "manager")
    proposer = _role_payload(models, "proposer")
    coder = _role_payload(models, "coder")
    return {
        "manager": manager,
        "proposer": proposer,
        "coder": coder,
        "claude_cli": manager,
        "claude_agents": {
            "automl-proposer": proposer,
            "automl-coder": coder,
        },
    }


def _experiment_route(
    *,
    project_name: str,
    experiment_id: str,
    dry_run: bool,
    namespace: str = "",
) -> str:
    from automl.mlflow import routing as mlflow_routing

    return mlflow_routing.experiment_route_for(
        project_name=project_name,
        experiment_id=experiment_id,
        namespace=namespace,
        dry_run=dry_run,
    )


def _session_args(
    *,
    project_root: Path,
    project_name: str,
    dry_run: bool,
    namespace: str,
    experiment_id: str = "",
) -> list[str]:
    args: list[str] = []
    if project_name:
        args.extend(["--project", project_name])
    args.extend(["--project-root", str(project_root)])
    if dry_run:
        args.append("--dry-run")
    if namespace:
        args.extend(["--namespace", namespace])
    if experiment_id:
        args.extend(["--experiment-id", experiment_id])
    return args


def _project_payload(project_root: Path, project_name: str) -> dict[str, str]:
    payload = {
        "root": str(project_root),
        "name": project_name,
        "package": "",
        "config_path": "",
        "instructions_path": "",
    }
    if project_name:
        payload.update(
            {
                "package": f"projects.{project_name}",
                "config_path": f"projects/{project_name}/config.py",
                "instructions_path": f"projects/{project_name}/PROJECT_INSTRUCTIONS.md",
            }
        )
    return payload


def _project_contract_payload(ctx: Any | None) -> dict[str, Any]:
    if ctx is None:
        return {
            "target_column": "",
            "raw_target_column": "",
            "primary_metric": "",
            "required_transformers": [],
        }
    try:
        from automl.model.preprocessing import describe_required_transformers

        required_transformers = describe_required_transformers(session=type("_Session", (), {"config": ctx})())
    except Exception:
        required_transformers = []
    return {
        "target_column": _text(getattr(ctx, "target_column", "")),
        "raw_target_column": _text(getattr(ctx, "raw_target_column", "")),
        "primary_metric": _text(getattr(ctx, "primary_metric", "")),
        "required_transformers": required_transformers,
    }


ROUTE_SEGMENT_RE = re.compile(r"[^A-Za-z0-9_.=-]+")
PACKAGE_NAME_RE = re.compile(r"^([A-Za-z0-9._-]+)")


def _route_segment(value: str) -> str:
    segment = ROUTE_SEGMENT_RE.sub("_", value.strip()).strip("._")
    return segment or "unknown"


def _proposal_handoff_path(route: str) -> str:
    path = Path(".cache") / "automl" / "tmp" / "proposals"
    route_segments = [_route_segment(item) for item in route.split("/") if item]
    for segment in route_segments or ["unrouted"]:
        path /= segment
    return str(path / "trial_proposal.json")


def _timeline_paths(route: str) -> dict[str, str]:
    path = Path(".cache") / "automl" / "tmp" / "timelines"
    route_segments = [_route_segment(item) for item in route.split("/") if item]
    for segment in route_segments or ["unrouted"]:
        path /= segment
    return {
        "agent_timeline": str(path / "agent_timeline.jsonl"),
        "agent_timeline_sessions": str(path / "sessions"),
    }


def _shell_command(parts: list[str]) -> str:
    return shlex.join(parts)


def _parse_package_name(spec: str) -> str:
    spec = spec.split("[", 1)[0]
    match = PACKAGE_NAME_RE.match(spec.strip())
    return match.group(1) if match else spec.strip()


def _load_allowed_dependencies(project_root: Path) -> list[str]:
    path = project_root / "pyproject.toml"
    if not path.exists():
        return []
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except Exception:
        return []
    deps: list[Any] = []
    project_deps = _mapping(data.get("project")).get("dependencies", [])
    if isinstance(project_deps, list):
        deps.extend(project_deps)
    groups = data.get("dependency-groups")
    if isinstance(groups, dict):
        for group_deps in groups.values():
            if isinstance(group_deps, list):
                deps.extend(group_deps)
    names: list[str] = []
    seen: set[str] = set()
    for dep in deps:
        if not isinstance(dep, str):
            continue
        name = _parse_package_name(dep)
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _error_context(
    *,
    project_root: Path,
    invocation: dict[str, Any],
    message: str,
) -> dict[str, Any]:
    payload_invocation = dict(invocation)
    payload_invocation["mode"] = "error"
    payload_invocation["error"] = message
    payload_invocation["needs_confirmation"] = False
    return {
        "schema_version": 1,
        "operation": "automl",
        "writes_during_render": False,
        "invocation": payload_invocation,
        "route": "",
        "execution_semantics": {},
        "paths": {},
        "project": {
            **_project_payload(project_root, ""),
            "experiment_id": "",
        },
        "project_contract": _project_contract_payload(None),
        "environment": {
            "allowed_dependencies": _load_allowed_dependencies(project_root),
        },
        "mlflow": {
            "tracking_uri": "",
        },
        "safe_commands": {},
    }


def _resolve_repo_root(project_root: Path) -> Path:
    """Resolve the repo root with the library's canonical walk-up.

    ``--project-root`` is ``${AUTOML_PROJECT_ROOT:-.}``: the launcher passes the
    repo root explicitly, but an interactive session passes its cwd, which may
    be anywhere inside the repo (e.g. ``projects/<name>/``). Walking up here is
    the same resolution every CLI verb gets via ``use_project``.
    """
    try:
        from automl.project import find_repo_root

        return find_repo_root(project_root)
    except Exception:
        return project_root.resolve()


def build_context(project_root: Path, arguments: str) -> dict[str, Any]:
    project_root = _resolve_repo_root(project_root)
    preflight = _load_preflight(Path(__file__).resolve().parent)
    invocation = preflight.parse_arguments(arguments)
    requested_project = _text(invocation.get("project")).strip()
    config_error = ""
    ctx = _load_project_context(project_root, requested_project)
    ctx_error = "" if ctx is not None else _load_project_context_error(project_root, requested_project)
    project_name = requested_project or (
        ctx.project_name if ctx is not None else _fallback_project_name(project_root, requested_project)
    )
    if ctx is not None and getattr(ctx, "run_config", None) is not None:
        experiment_id = _text(ctx.run_config.experiment_id)
    else:
        experiment_id = ""
        if ctx is not None and invocation.get("mode") == "run":
            config_error = f"RUN_CONFIG missing from {ctx.config_path}"
    namespace = _text(invocation.get("namespace")).strip()
    tracking_uri = _text(
        ctx.mlflow_tracking_uri
        if ctx is not None
        else _env_value(project_root, "MLFLOW_TRACKING_URI")
    )
    dry_run = bool(invocation.get("dry_run"))
    route = ""
    if project_name and experiment_id:
        route = _experiment_route(
            project_name=project_name,
            experiment_id=experiment_id,
            dry_run=dry_run,
            namespace=namespace,
        )
    if invocation.get("mode") == "run":
        if config_error:
            return _error_context(
                project_root=project_root,
                invocation=invocation,
                message=config_error,
            )
        if ctx_error:
            return _error_context(
                project_root=project_root,
                invocation=invocation,
                message=ctx_error,
            )
        if not project_name or not experiment_id:
            return _error_context(
                project_root=project_root,
                invocation=invocation,
                message=(
                    "active project and experiment_id are required; pass "
                    "--project <project_name> or run from a configured project"
                ),
            )

    repo_root = Path(__file__).resolve().parents[3]
    repo_hooks_dir = repo_root / "hooks"
    session_args = _session_args(
        project_root=project_root,
        project_name=project_name,
        dry_run=dry_run,
        namespace=namespace,
        experiment_id=experiment_id,
    )
    loop_context_args = [
        "uv",
        "run",
        "automl",
        *session_args,
        "experiment",
        "proposer-context",
    ]
    materialize_dataset_args = [
        "uv",
        "run",
        "automl",
        *session_args,
        "data",
        "materialize",
    ]
    if invocation.get("refresh_data"):
        materialize_dataset_args.append("--refresh-data")
    if invocation.get("refresh_source"):
        materialize_dataset_args.append("--refresh-source")

    proposal_handoff_path = _proposal_handoff_path(route)
    timeline_paths = _timeline_paths(route)
    create_trial_args = [
        "uv",
        "run",
        "automl",
        *session_args,
        "trial",
        "create",
        "--proposal-json",
        proposal_handoff_path,
    ]
    allowed_dependencies = _load_allowed_dependencies(project_root)
    timeline_publish_args = [
        "uv",
        "run",
        str(repo_hooks_dir / "agent_timeline.py"),
        "--project-root",
        str(project_root),
    ]
    if project_name:
        timeline_publish_args.extend(["--project", project_name])
    timeline_publish_args.append("publish")
    session_id = (
        os.environ.get("AUTOML_SESSION_ID")
        or os.environ.get("CLAUDE_SESSION_ID")
        or ""
    )
    if session_id:
        timeline_publish_args.extend(["--session-id", session_id])

    return {
        "schema_version": 1,
        "operation": "automl",
        "writes_during_render": False,
        "invocation": invocation,
        "user_instructions": invocation.get("user_instructions") or [],
        "route": route,
        "execution_semantics": {
            "dry_run_is_proposal_only": False,
            "creates_trial_directories": True,
            "dispatches_coder": True,
            "executes_trial_runner": True,
            "logs_mlflow_trial_runs": True,
            "mlflow_route": route,
            "run_mode": "dry_run" if dry_run else "full_run",
        },
        "paths": {
            "proposal_handoff": proposal_handoff_path,
            **timeline_paths,
        },
        "project": {
            **_project_payload(project_root, project_name),
            "experiment_id": experiment_id,
        },
        "project_contract": _project_contract_payload(ctx),
        "models": _models_payload({"models": ctx.models} if ctx is not None else {}),
        "environment": {
            "allowed_dependencies": allowed_dependencies,
        },
        # Routing truth only. The active dataset comes from running
        # safe_commands.materialize_dataset; the current MLflow summary comes
        # from safe_commands.loop_context. Neither is known at render time.
        "mlflow": {
            "tracking_uri": tracking_uri,
        },
        "safe_commands": {
            "validate": _shell_command(
                [
                    "uv",
                    "run",
                    str(Path(__file__).resolve()),
                    "--project-root",
                    str(project_root),
                    "--arguments",
                    f"experiment run --project {project_name} --dry-run --max-iter 1"
                    if project_name
                    else "experiment run --dry-run --max-iter 1",
                ]
            ),
            "loop_context": _shell_command(loop_context_args),
            "materialize_dataset": _shell_command(materialize_dataset_args),
            "persist_proposal": _shell_command(
                [
                    "uv",
                    "run",
                    "automl",
                    *session_args,
                    "validate",
                    "proposal",
                    "--proposal-json",
                    "-",
                    "--output",
                    proposal_handoff_path,
                ]
            ),
            "create_trial": _shell_command(create_trial_args),
            "validate_proposal": (
                _shell_command(
                    [
                        "uv",
                        "run",
                        "automl",
                        *session_args,
                        "validate",
                        "proposal",
                        "--proposal-json",
                        "<proposal.json>",
                    ]
                )
            ),
            "timeline_publish": _shell_command(timeline_publish_args),
        },
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
