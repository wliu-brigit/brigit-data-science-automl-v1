# Serving-Validation Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A validation timeout or signal-killed subprocess always produces a JSON-serializable, labeled failure report — never crashes the trial — and the timeout is a `RunConfig` knob (default 300s) instead of a hardcoded 120s.

**Architecture:** Three contained changes in `automl/runner/serving_validation.py` (decode helper, configured timeout, signal-exit guard) plus one new validated field on `RunConfig`. Trial status semantics unchanged (design §3: fail-soft; `validation.status` is the deployability signal).

**Tech Stack:** Python 3.13, pytest (`uv run pytest`), stdlib `subprocess`/`signal`.

**Design:** `docs/execution/trial-reliability/design.md` §6.

---

### Task 1: `RunConfig.serving_validation_seconds`

**Files:**
- Modify: `automl/project/run_config.py` (the `RunConfig` dataclass, ~line 127)
- Test: `tests/unit/project/test_run_config.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/project/test_run_config.py` (reuse the file's existing imports/helpers if equivalent ones exist; otherwise add):

```python
import pytest

from automl.project.run_config import ModelRoute, ModelsConfig, RunConfig


def _models() -> ModelsConfig:
    route = ModelRoute(model="claude-test", effort="low")
    return ModelsConfig(manager=route, proposer=route, coder=route)


def test_serving_validation_seconds_defaults_to_300():
    config = RunConfig(experiment_id="exp", models=_models(), per_trial_seconds=600)
    assert config.serving_validation_seconds == 300


def test_serving_validation_seconds_accepts_override():
    config = RunConfig(
        experiment_id="exp",
        models=_models(),
        per_trial_seconds=600,
        serving_validation_seconds=42,
    )
    assert config.serving_validation_seconds == 42


@pytest.mark.parametrize("bad", [0, -5, True, 1.5, "300"])
def test_serving_validation_seconds_rejects_invalid(bad):
    with pytest.raises(ValueError):
        RunConfig(
            experiment_id="exp",
            models=_models(),
            per_trial_seconds=600,
            serving_validation_seconds=bad,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/project/test_run_config.py -q -k serving_validation`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'serving_validation_seconds'` (and the default test fails on the missing attribute).

- [ ] **Step 3: Implement the field**

In `automl/project/run_config.py`, `RunConfig`: add the field declaration after `eval_split: str`:

```python
    serving_validation_seconds: int
```

Add the keyword parameter to `__init__` after `eval_split: str = "test"`:

```python
        serving_validation_seconds: int = 300,
```

Add validation next to the `per_trial_seconds` check (same pattern — `bool` is an `int` subclass, reject it explicitly):

```python
        if (
            isinstance(serving_validation_seconds, bool)
            or not isinstance(serving_validation_seconds, int)
            or serving_validation_seconds < 1
        ):
            raise ValueError(
                "serving_validation_seconds must be a positive integer, "
                f"got {serving_validation_seconds!r}"
            )
```

And the assignment next to the others:

```python
        object.__setattr__(self, "serving_validation_seconds", serving_validation_seconds)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/project/test_run_config.py -q`
Expected: all PASS (new and pre-existing).

- [ ] **Step 5: Commit**

```bash
git add automl/project/run_config.py tests/unit/project/test_run_config.py
git commit -m "feat(run-config): serving_validation_seconds knob (default 300)"
```

---

### Task 2: `_decode_tail` — the timeout handler must serialize

**Files:**
- Modify: `automl/runner/serving_validation.py` (the `TimeoutExpired` handler, ~line 533)
- Test: `tests/unit/runner/test_validation_errors.py` (append)

**Context for the implementer:** on `TimeoutExpired`, `exc.stderr` is `bytes` even though `subprocess.run` was called with `text=True` (CPython quirk). The current handler slices the bytes but never decodes (`serving_validation.py:536`), so `json.dumps(report)` raises `TypeError` — the handler whose comment says "don't crash the trial" crashes the trial. This is the bug that halted the neobank loop (see `../finding-timeout-crash-and-halt.md`).

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/runner/test_validation_errors.py`:

```python
import json
import subprocess
from pathlib import Path

from automl.runner import serving_validation


class _FakeConfig:
    mlflow_tracking_uri = "http://127.0.0.1:9"
    repo_root = Path(".")
    run_config = None


class _FakeSession:
    config = _FakeConfig()


def test_timeout_report_serializes_with_bytes_stderr(tmp_path, monkeypatch):
    def _raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=["python"], timeout=7, stderr=b"\xff boom \xfe"
        )

    monkeypatch.setattr(serving_validation.subprocess, "run", _raise_timeout)
    report_path = tmp_path / "report.json"
    report = serving_validation._run_pyfunc_validation(
        run_id="run123",
        active=_FakeSession(),
        input_parquet=tmp_path / "input.parquet",
        input_csv=tmp_path / "input.csv",
        expected_parquet=tmp_path / "expected.parquet",
        input_schema=tmp_path / "input_schema.json",
        report_path=report_path,
        tolerance=1e-10,
    )
    assert report["status"] == "failed"
    assert report["error_class"] == "TimeoutExpired"
    assert isinstance(report["stderr_tail"], str)
    # The whole point: the report written to disk is valid JSON.
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "failed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/runner/test_validation_errors.py -q -k bytes_stderr`
Expected: FAIL with `TypeError: Object of type bytes is not JSON serializable`.

- [ ] **Step 3: Implement `_decode_tail` and use it**

In `automl/runner/serving_validation.py`, add a module-level helper (near the other `_`-helpers):

```python
def _decode_tail(raw: object, *, limit: int = 1000) -> str:
    """Last ``limit`` chars of subprocess output, always as ``str``.

    ``TimeoutExpired.stderr`` is bytes even under ``text=True``.
    """
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw[-limit:].decode("utf-8", "replace")
    return str(raw)[-limit:]
```

Replace the handler's stderr line:

```python
        stderr_tail = (exc.stderr or b"")[-1000:] if isinstance(exc.stderr, bytes) else (exc.stderr or "")[-1000:]
```

with:

```python
        stderr_tail = _decode_tail(exc.stderr)
```

Also route the two non-timeout stderr tails through it (lines ~564 and ~570): replace `completed.stderr[-1000:]` with `_decode_tail(completed.stderr)` in both places.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/runner/test_validation_errors.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add automl/runner/serving_validation.py tests/unit/runner/test_validation_errors.py
git commit -m "fix(validation): decode subprocess output tails; timeout report always serializes"
```

---

### Task 3: Timeout comes from `RunConfig`

**Files:**
- Modify: `automl/runner/serving_validation.py` (module constant ~line 25; `log_validation_artifacts`; `_run_pyfunc_validation` signature)
- Test: `tests/unit/runner/test_validation_errors.py` (append)

- [ ] **Step 1: Write the failing test**

```python
def test_validation_timeout_uses_run_config(tmp_path, monkeypatch):
    captured = {}

    def _capture_run(*args, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        raise subprocess.TimeoutExpired(cmd=["python"], timeout=kwargs.get("timeout"))

    monkeypatch.setattr(serving_validation.subprocess, "run", _capture_run)

    class _RunConfig:
        serving_validation_seconds = 77

    class _Config(_FakeConfig):
        run_config = _RunConfig()

    class _Session:
        config = _Config()

    report = serving_validation._run_pyfunc_validation(
        run_id="run123",
        active=_Session(),
        input_parquet=tmp_path / "input.parquet",
        input_csv=tmp_path / "input.csv",
        expected_parquet=tmp_path / "expected.parquet",
        input_schema=tmp_path / "input_schema.json",
        report_path=tmp_path / "report.json",
        tolerance=1e-10,
    )
    assert captured["timeout"] == 77
    assert "77" in report["error"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/runner/test_validation_errors.py -q -k uses_run_config`
Expected: FAIL — `captured["timeout"] == 120` (the hardcoded constant).

- [ ] **Step 3: Implement**

In `automl/runner/serving_validation.py`:

Replace the constant block (keep the comment's intent, update the value story):

```python
# Fallback wall-clock budget for the serving-validation subprocess when the
# session has no RUN_CONFIG. Projects tune this via
# RUN_CONFIG.serving_validation_seconds (default 300 — the observed full-data
# baseline sat at 120.07s, so the old 120s cap was boundary-tight).
_DEFAULT_VALIDATION_TIMEOUT_S = 300
```

Add a resolver helper:

```python
def _validation_timeout_seconds(active: Session) -> int:
    run_config = getattr(active.config, "run_config", None)
    value = getattr(run_config, "serving_validation_seconds", None)
    return int(value) if value else _DEFAULT_VALIDATION_TIMEOUT_S
```

In `_run_pyfunc_validation`, compute `timeout_s = _validation_timeout_seconds(active)` at the top, pass `timeout=timeout_s` to `subprocess.run`, and update the timeout report's error message to reference it:

```python
            "error": (
                f"validation subprocess exceeded {timeout_s}s "
                f"loading/benchmarking {resolved_model_uri!r} "
                "(RUN_CONFIG.serving_validation_seconds)"
            ),
```

Remove every remaining reference to `_VALIDATION_TIMEOUT_S` (grep: `grep -n _VALIDATION_TIMEOUT_S automl/ -r`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/runner/test_validation_errors.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add automl/runner/serving_validation.py tests/unit/runner/test_validation_errors.py
git commit -m "feat(validation): timeout from RUN_CONFIG.serving_validation_seconds"
```

---

### Task 4: Signal-exit guard — a signal-killed child gets a labeled report

**Files:**
- Modify: `automl/runner/serving_validation.py` (after the `subprocess.run` call returns, before the `report_path.exists()` check, ~line 553)
- Test: `tests/unit/runner/test_validation_errors.py` (append)

**Context:** a child killed by a signal gives `completed.returncode < 0` on POSIX. Today it falls into the generic missing-report fallback (often with an empty stderr tail). A signal-killed child's report file, if present, is untrustworthy (it may have died mid-write), so the guard synthesizes a labeled report unconditionally on a signal exit.

- [ ] **Step 1: Write the failing test**

```python
def test_signal_killed_child_produces_labeled_report(tmp_path, monkeypatch):
    def _fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0] if args else [], returncode=-11, stdout="", stderr=""
        )

    monkeypatch.setattr(serving_validation.subprocess, "run", _fake_run)
    report_path = tmp_path / "report.json"
    report = serving_validation._run_pyfunc_validation(
        run_id="run123",
        active=_FakeSession(),
        input_parquet=tmp_path / "input.parquet",
        input_csv=tmp_path / "input.csv",
        expected_parquet=tmp_path / "expected.parquet",
        input_schema=tmp_path / "input_schema.json",
        report_path=report_path,
        tolerance=1e-10,
    )
    assert report["status"] == "failed"
    assert report["error_class"] == "SignalExit"
    assert report["signal"] == 11
    assert report["signal_name"] == "SIGSEGV"
    assert "SIGSEGV" in report["error"]
    assert json.loads(report_path.read_text(encoding="utf-8"))["error_class"] == "SignalExit"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/runner/test_validation_errors.py -q -k signal_killed`
Expected: FAIL — `KeyError: 'error_class'` (the generic fallback report has no such key).

- [ ] **Step 3: Implement the guard**

Add `import signal` to the module imports. In `_run_pyfunc_validation`, immediately after the `try/except TimeoutExpired` block (i.e. once `completed` exists), insert before the `if report_path.exists():` line:

```python
    if completed.returncode < 0:
        signum = -completed.returncode
        try:
            signal_name = signal.Signals(signum).name
        except ValueError:
            signal_name = f"signal {signum}"
        report = {
            "schema_version": 1,
            "status": "failed",
            "model_uri": resolved_model_uri,
            "row_count": 0,
            "max_abs_diff": None,
            "tolerance": tolerance,
            "error": (
                f"validation subprocess died on {signal_name} "
                f"(returncode {completed.returncode}); any partial report is untrusted"
            ),
            "error_class": "SignalExit",
            "signal": signum,
            "signal_name": signal_name,
            "stderr_tail": _decode_tail(completed.stderr),
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
```

- [ ] **Step 4: Run the full runner + project unit suites**

Run: `uv run pytest tests/unit/runner tests/unit/project -q`
Expected: all PASS.

- [ ] **Step 5: Run the contract suite (doc/shape ratchets)**

Run: `uv run pytest tests/contracts -q`
Expected: PASS. If a docs-truth contract fails on a phrase this plan changed, update that pinned phrase in the same commit — that's the contract tests' purpose.

- [ ] **Step 6: Commit**

```bash
git add automl/runner/serving_validation.py tests/unit/runner/test_validation_errors.py
git commit -m "feat(validation): labeled SignalExit report for signal-killed validation child"
```

---

## Done criteria

- `uv run pytest tests/unit tests/contracts -q` green.
- `grep -rn "_VALIDATION_TIMEOUT_S" automl/` returns nothing (replaced by `_DEFAULT_VALIDATION_TIMEOUT_S` + config).
- Behavior unchanged for passing validations; failing ones always leave a serializable, labeled `validation/report.json` + `validation.status` tag.
