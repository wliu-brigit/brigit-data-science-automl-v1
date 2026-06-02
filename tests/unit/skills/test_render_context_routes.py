from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]


def _render_context_module():
    module_path = REPO_ROOT / "agent-skills" / "skills" / "automl" / "scripts" / "render_context.py"
    spec = importlib.util.spec_from_file_location("automl_skill_render_context", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("arguments", "route"),
    [
        (
            "experiment run --project example_homecredit --max-iter 1",
            "example_homecredit/example-homecredit",
        ),
        (
            "experiment run --project example_homecredit --dry-run --max-iter 1",
            "dry_run/example_homecredit/example-homecredit",
        ),
        (
            "experiment run --project example_homecredit --namespace qa --dry-run --max-iter 1",
            "qa/dry_run/example_homecredit/example-homecredit",
        ),
    ],
)
def test_render_context_pins_route_scoped_cache_paths(
    arguments,
    route,
):
    module = _render_context_module()
    proposal_path = str(
        Path(".cache") / "automl" / "tmp" / "proposals" / route / "trial_proposal.json"
    )
    timeline_root = Path(".cache") / "automl" / "tmp" / "timelines" / route

    context = module.build_context(REPO_ROOT, arguments)

    assert context["route"] == route
    assert context["execution_semantics"]["mlflow_route"] == route
    assert context["paths"] == {
        "proposal_handoff": proposal_path,
        "agent_timeline": str(timeline_root / "agent_timeline.jsonl"),
        "agent_timeline_sessions": str(timeline_root / "sessions"),
    }
    assert proposal_path in context["safe_commands"]["persist_proposal"]
    assert proposal_path in context["safe_commands"]["create_trial"]
