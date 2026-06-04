from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_automl_preflight_import_stays_lightweight():
    code = f"""
import importlib.util
import json
import sys
from pathlib import Path

repo_root = Path({str(REPO_ROOT)!r})
module_path = repo_root / "agent-skills" / "skills" / "automl" / "scripts" / "preflight.py"
spec = importlib.util.spec_from_file_location("automl_skill_preflight", module_path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

blocked = [
    "mlflow",
    "automl.mlflow",
    "automl.data",
    "automl.agent.launch",
    "automl.agent.proposer_context",
]
loaded = [name for name in blocked if name in sys.modules]
if loaded:
    print(json.dumps({{"loaded": loaded}}, indent=2))
    raise SystemExit(1)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"


def test_agent_facade_public_exports_still_resolve():
    import automl.agent as agent
    from automl.agent import (
        DISALLOWED,
        Proposal,
        build_launch,
        gather_proposer_context,
        handle_event,
        publish,
        validate_proposal,
    )

    assert agent.__all__ == [
        "DISALLOWED",
        "Proposal",
        "build_launch",
        "gather_proposer_context",
        "handle_event",
        "publish",
        "validate_proposal",
    ]
    assert DISALLOWED
    assert Proposal.__name__ == "Proposal"
    assert build_launch.__name__ == "build_launch"
    assert gather_proposer_context.__name__ == "gather_proposer_context"
    assert handle_event.__name__ == "handle_event"
    assert publish.__name__ == "publish"
    assert validate_proposal.__name__ == "validate_proposal"
