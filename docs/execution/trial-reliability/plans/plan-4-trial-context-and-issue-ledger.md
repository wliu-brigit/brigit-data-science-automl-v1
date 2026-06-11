# TrialContext + Issue Ledger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A trial that finished can still report everything that went wrong inside it: best-effort failures stop being swallowed silently and land in a crash-safe local ledger, published to MLflow as `trial/issues.json` + a `trial.issue_count` tag on both exit paths.

**Architecture:** Design §7 — *separate the record from the machinery*. A new `IssueRecorder` (the record) and `TrialContext` (one object composing identity + `TimingRecorder` + `IssueRecorder` + trial dir) thread through the runner in place of N loose parameters. The straight-line `_run_trial` machinery is unchanged in shape; the future modular/retry runner inherits these published schemas untouched. Land this plan **last** — it has the widest diff and touches the failure path the agent loop depends on.

**Tech Stack:** Python 3.13, pytest via `uv run`, stdlib `faulthandler`/`json`.

**Design:** `docs/execution/trial-reliability/design.md` §7; boundary #3 (parent-crash durability) is explicitly out of scope — parked at `docs/to-do/runner-crash-supervision.md`.

---

### Task 1: `IssueRecorder` (`automl/runner/issues.py`)

**Files:**
- Create: `automl/runner/issues.py`
- Test: `tests/unit/runner/test_issues.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/runner/test_issues.py`:

```python
from __future__ import annotations

import json

import pytest

from automl.runner.issues import IssueRecorder

pytestmark = pytest.mark.unit


def test_record_exception_captures_class_message_and_phase():
    recorder = IssueRecorder()
    try:
        raise ValueError("boom")
    except ValueError as exc:
        recorder.record(exc, phase="evaluation", severity="warning")
    (issue,) = recorder.snapshot()
    assert issue["phase"] == "evaluation"
    assert issue["severity"] == "warning"
    assert issue["error_class"] == "ValueError"
    assert issue["message"] == "boom"
    assert issue["traceback_tail"]
    assert recorder.count == 1


def test_record_plain_message():
    recorder = IssueRecorder()
    recorder.record("latency not measured", phase="validation_publish")
    (issue,) = recorder.snapshot()
    assert issue["severity"] == "error"
    assert issue["error_class"] == ""
    assert issue["message"] == "latency not measured"


def test_jsonl_appended_as_events_happen(tmp_path):
    jsonl = tmp_path / "issues.jsonl"
    recorder = IssueRecorder(jsonl_path=jsonl)
    recorder.record("first", phase="fit")
    recorder.record("second", phase="evaluation")
    lines = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    assert [line["message"] for line in lines] == ["first", "second"]


def test_recording_never_raises_when_jsonl_unwritable(tmp_path):
    recorder = IssueRecorder(jsonl_path=tmp_path / "no-such-dir" / "issues.jsonl")
    recorder.record("still recorded in memory", phase="fit")
    assert recorder.count == 1


def test_snapshot_is_json_serializable():
    recorder = IssueRecorder()
    try:
        raise RuntimeError("x")
    except RuntimeError as exc:
        recorder.record(exc, phase="fit")
    json.dumps(recorder.snapshot())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/runner/test_issues.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'automl.runner.issues'`.

- [ ] **Step 3: Implement `IssueRecorder`**

Create `automl/runner/issues.py`:

```python
"""Trial issue ledger: the durable record of what went wrong mid-trial.

Best-effort steps record here instead of swallowing exceptions. Events are
appended to a local JSONL as they happen (crash-safe: a native crash of the
runner still leaves the file on disk) and published to MLflow at trial end by
``automl.runner.issue_artifacts``. Recording must never raise — the ledger
cannot be allowed to become a failure source itself.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from automl.runner.failures import ExceptionSnapshot


class IssueRecorder:
    def __init__(self, jsonl_path: Path | None = None) -> None:
        self._issues: list[dict[str, Any]] = []
        self._jsonl_path = jsonl_path

    def record(
        self,
        problem: BaseException | str,
        *,
        phase: str,
        severity: str = "error",
    ) -> None:
        if isinstance(problem, BaseException):
            snapshot = ExceptionSnapshot.from_exception(problem)
            error_class = snapshot.error_class
            message = snapshot.message
            traceback_tail = snapshot.to_dict()["traceback_tail"]
        else:
            error_class = ""
            message = str(problem)
            traceback_tail = []
        entry = {
            "at": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "severity": severity,
            "error_class": error_class,
            "message": message,
            "traceback_tail": traceback_tail,
        }
        self._issues.append(entry)
        self._append_jsonl(entry)

    def snapshot(self) -> list[dict[str, Any]]:
        return [dict(entry) for entry in self._issues]

    @property
    def count(self) -> int:
        return len(self._issues)

    def _append_jsonl(self, entry: dict[str, Any]) -> None:
        if self._jsonl_path is None:
            return
        try:
            with self._jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
        except OSError:
            # Ledger writes are best-effort by definition; the in-memory
            # record still publishes at trial end.
            pass


__all__ = ["IssueRecorder"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/runner/test_issues.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add automl/runner/issues.py tests/unit/runner/test_issues.py
git commit -m "feat(runner): IssueRecorder — crash-safe trial issue ledger"
```

---

### Task 2: `TrialContext` (`automl/runner/context.py`)

**Files:**
- Create: `automl/runner/context.py`
- Test: `tests/unit/runner/test_context.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/runner/test_context.py`:

```python
from __future__ import annotations

import pytest

from automl.runner.context import TrialContext

pytestmark = pytest.mark.unit


def test_phase_delegates_to_timing():
    ctx = TrialContext()
    with ctx.phase("fit"):
        pass
    assert "fit" in ctx.timing.phases
    assert ctx.timing.last_phase == "fit"


def test_record_issue_defaults_phase_to_last_timing_phase():
    ctx = TrialContext()
    with ctx.phase("evaluation"):
        pass
    ctx.record_issue("went sideways", severity="warning")
    (issue,) = ctx.issues.snapshot()
    assert issue["phase"] == "evaluation"
    assert issue["severity"] == "warning"


def test_record_issue_with_explicit_phase():
    ctx = TrialContext()
    ctx.record_issue("early problem", phase="model_import")
    (issue,) = ctx.issues.snapshot()
    assert issue["phase"] == "model_import"


def test_jsonl_lands_in_trial_dir(tmp_path):
    ctx = TrialContext(trial_dir=tmp_path)
    ctx.record_issue("evidence", phase="fit")
    assert (tmp_path / "issues.jsonl").exists()


def test_identity_fields_fill_in_as_known():
    ctx = TrialContext()
    ctx.run_id = "run1"
    ctx.trial_id = "1_slug"
    ctx.trial_number = 1
    ctx.slug = "slug"
    ctx.strategy = "baseline"
    assert ctx.run_id == "run1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/runner/test_context.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'automl.runner.context'`.

- [ ] **Step 3: Implement `TrialContext`**

Create `automl/runner/context.py`:

```python
"""Trial-scoped context: one home for the runner's cross-cutting state.

Composes the trial's identity (filled in as it becomes known), the
``TimingRecorder``, and the ``IssueRecorder`` so one object threads through
the runner instead of N loose parameters. This is the *record* side of the
record/machinery split (design: trial-reliability §7) — a future step-based
runner passes this same context to each step.
"""

from __future__ import annotations

from pathlib import Path

from automl.runner.issues import IssueRecorder
from automl.runner.timing import TimingRecorder

ISSUES_JSONL_NAME = "issues.jsonl"


class TrialContext:
    def __init__(self, *, trial_dir: Path | None = None) -> None:
        self.run_id: str = ""
        self.trial_id: str = ""
        self.trial_number: int | None = None
        self.slug: str = ""
        self.strategy: str = ""
        self.trial_dir = trial_dir
        self.timing = TimingRecorder()
        # Memory-only when there is no trial dir to leave evidence in
        # (project-run trials); the published record still lands either way.
        self.issues = IssueRecorder(
            jsonl_path=(trial_dir / ISSUES_JSONL_NAME) if trial_dir else None
        )

    def phase(self, name: str):
        return self.timing.phase(name)

    def record_issue(
        self,
        problem: BaseException | str,
        *,
        phase: str | None = None,
        severity: str = "error",
    ) -> None:
        resolved = phase or self.timing.last_phase or "unknown"
        self.issues.record(problem, phase=resolved, severity=severity)


__all__ = ["ISSUES_JSONL_NAME", "TrialContext"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/runner/test_context.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add automl/runner/context.py tests/unit/runner/test_context.py
git commit -m "feat(runner): TrialContext — one home for trial identity, timing, issues"
```

---

### Task 3: Publishing (`automl/runner/issue_artifacts.py` + tag constant)

**Files:**
- Create: `automl/runner/issue_artifacts.py`
- Modify: `automl/mlflow/tags.py` (add `TRIAL_ISSUE_COUNT = "trial.issue_count"` next to the other `TRIAL_*` constants, and `"TRIAL_ISSUE_COUNT"` to `__all__`)
- Modify: `automl/runner/artifacts.py` (re-export `log_issue_artifacts`, matching how `log_validation_artifacts` is re-exported)
- Test: `tests/unit/runner/test_issue_artifacts.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/runner/test_issue_artifacts.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from automl.mlflow import tags as mlflow_tags
from automl.runner.context import TrialContext
from automl.runner import issue_artifacts

pytestmark = pytest.mark.unit


def test_publishes_issues_json_and_count_tag(monkeypatch):
    published = {}

    def fake_write_local_file(run_id, artifact_path, local_path):
        published["run_id"] = run_id
        published["artifact_path"] = artifact_path
        published["payload"] = json.loads(Path(local_path).read_text(encoding="utf-8"))

    tags_set = {}
    monkeypatch.setattr(
        issue_artifacts.runner_artifacts, "write_local_file", fake_write_local_file
    )
    monkeypatch.setattr(
        issue_artifacts.mlflow_trial,
        "set_tags",
        lambda run_id, tags: tags_set.update(tags),
    )

    ctx = TrialContext()
    ctx.record_issue("something best-effort failed", phase="evaluation", severity="warning")
    issue_artifacts.log_issue_artifacts("run123", ctx.issues)

    assert published["run_id"] == "run123"
    assert published["artifact_path"] == "trial/issues.json"
    assert published["payload"]["schema_version"] == 1
    assert published["payload"]["issues"][0]["message"] == "something best-effort failed"
    assert tags_set[mlflow_tags.TRIAL_ISSUE_COUNT] == 1


def test_no_run_id_is_a_noop(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("must not publish without a run")

    monkeypatch.setattr(issue_artifacts.runner_artifacts, "write_local_file", explode)
    ctx = TrialContext()
    ctx.record_issue("x", phase="fit")
    issue_artifacts.log_issue_artifacts("", ctx.issues)


def test_zero_issues_still_publishes_count_zero(monkeypatch):
    published = {}
    tags_set = {}
    monkeypatch.setattr(
        issue_artifacts.runner_artifacts,
        "write_local_file",
        lambda run_id, artifact_path, local_path: published.update(
            payload=json.loads(Path(local_path).read_text(encoding="utf-8"))
        ),
    )
    monkeypatch.setattr(
        issue_artifacts.mlflow_trial,
        "set_tags",
        lambda run_id, tags: tags_set.update(tags),
    )
    issue_artifacts.log_issue_artifacts("run123", TrialContext().issues)
    assert published["payload"]["issues"] == []
    assert tags_set[mlflow_tags.TRIAL_ISSUE_COUNT] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/runner/test_issue_artifacts.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'automl.runner.issue_artifacts'`.

- [ ] **Step 3: Implement**

Add to `automl/mlflow/tags.py` next to the other `TRIAL_*` constants:

```python
TRIAL_ISSUE_COUNT = "trial.issue_count"
```

(and `"TRIAL_ISSUE_COUNT"` in `__all__`.)

Create `automl/runner/issue_artifacts.py` (same shape as `failure_artifacts.py`):

```python
"""Trial issue-ledger publishing helpers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from automl.mlflow import tags as mlflow_tags
from automl.mlflow import trial as mlflow_trial
from automl.mlflow.trial.artifacts import runner as runner_artifacts
from automl.runner.issues import IssueRecorder

ISSUES_ARTIFACT = "trial/issues.json"


def log_issue_artifacts(run_id: str, issues: IssueRecorder) -> None:
    """Publish the ledger + count tag. Called on BOTH trial exit paths."""
    if not run_id:
        return
    payload = {"schema_version": 1, "issues": issues.snapshot()}
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "issues.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        runner_artifacts.write_local_file(run_id, ISSUES_ARTIFACT, path)
    mlflow_trial.set_tags(run_id, {mlflow_tags.TRIAL_ISSUE_COUNT: issues.count})


__all__ = ["ISSUES_ARTIFACT", "log_issue_artifacts"]
```

In `automl/runner/artifacts.py`, add the re-export alongside the existing ones:

```python
from automl.runner.issue_artifacts import log_issue_artifacts
```

(and add `"log_issue_artifacts"` to its `__all__` if the module defines one.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/runner/test_issue_artifacts.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add automl/mlflow/tags.py automl/runner/issue_artifacts.py automl/runner/artifacts.py tests/unit/runner/test_issue_artifacts.py
git commit -m "feat(runner): publish trial/issues.json + trial.issue_count tag"
```

---

### Task 4: Thread `TrialContext` through `_run_trial`

**Files:**
- Modify: `automl/runner/trial.py` (mechanical: `timing` → `ctx`; collapse `_publish_failure_artifacts` params; publish ledger on both exit paths; convert `_try_log_train_eval`; `faulthandler.enable()`)
- Test: existing `tests/unit/runner/` suite + one new test in `tests/unit/runner/test_trial_data_contract.py`-style fashion is NOT needed; the conversion is covered by `test_trial_folder_execution.py` plus the new test below.

- [ ] **Step 1: Write the failing test for the converted best-effort site**

Append to `tests/unit/runner/test_issues.py`:

```python
def test_try_log_train_eval_records_issue_instead_of_swallowing(monkeypatch):
    from automl.runner import trial as trial_module
    from automl.runner.context import TrialContext

    def explode(**kwargs):
        raise RuntimeError("train eval blew up")

    monkeypatch.setattr(trial_module, "prepare_eval_dataset", explode)
    ctx = TrialContext()
    trial_module._try_log_train_eval(
        ctx=ctx,
        run_id="run1",
        active=object(),
        model=object(),
        dataset_id="ds_001",
        train_split="train",
        feature_registry=object(),
    )
    (issue,) = ctx.issues.snapshot()
    assert issue["severity"] == "warning"
    assert issue["error_class"] == "RuntimeError"
    assert "train eval blew up" in issue["message"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/runner/test_issues.py -q -k try_log_train_eval`
Expected: FAIL — `TypeError: _try_log_train_eval() got an unexpected keyword argument 'ctx'`.

- [ ] **Step 3: Convert `trial.py` (mechanical sweep)**

In `automl/runner/trial.py`:

1. Imports: add `import faulthandler`, `from automl.runner.context import TrialContext`, and extend the `.artifacts` import with `log_issue_artifacts`.
2. In `run_trial(...)`, first line of the body: `faulthandler.enable()` (idempotent; makes any native crash of the runner process dump a traceback to stderr).
3. In `_run_trial`, replace the loose state with the context. Delete `run_id = ""`, `trial_number = None`, `trial_id = ""`, `slug = ""`, `strategy = ""`, `timing = TimingRecorder()` and create instead:

```python
    ctx = TrialContext(trial_dir=context.trial_dir)
```

4. Mechanical replacements throughout `_run_trial`:
   - `with timing.phase(` → `with ctx.phase(`
   - `run_id = active_run_id` → `ctx.run_id = active_run_id` (and so on for every assignment to `trial_number`, `trial_id`, `slug`, `strategy` — assign to the `ctx.` field)
   - every later *read* of those locals (`run_id=run_id`, `trial_id=trial_id`, f-strings, `TrialResult(...)` fields) → the `ctx.` field
   - `timing=timing` in the `log_validation_artifacts(...)` call → `timing=ctx.timing`
   - `build_runner_timing_summary(timing.snapshot())` → `build_runner_timing_summary(ctx.timing.snapshot())`
5. Success path: immediately after `log_manifest(...)` and before `return TrialResult(...)`, add:

```python
            log_issue_artifacts(ctx.run_id, ctx.issues)
```

6. Failure path: change `_publish_failure_artifacts` to take the context:

```python
def _publish_failure_artifacts(
    *,
    ctx: TrialContext,
    active: Session,
    exc: BaseException,
) -> None:
    has_agent_proposal = log_agent_proposal(run_id=ctx.run_id, trial_dir=ctx.trial_dir)
    log_failure_artifacts(
        failure=RunnerFailureReport(
            runner_kind="trial",
            phase=ctx.timing.last_phase or "unknown",
            exception=ExceptionSnapshot.from_exception(exc),
            run_id=ctx.run_id,
            project_name=active.project_name,
            experiment_id=active.active_experiment_id,
            trial_id=ctx.trial_id,
            trial_number=ctx.trial_number,
            trial_slug=ctx.slug,
            trial_strategy=ctx.strategy,
            trial_dir=ctx.trial_dir,
            timing=ctx.timing.snapshot(),
        ),
        has_agent_proposal=has_agent_proposal,
    )
    log_issue_artifacts(ctx.run_id, ctx.issues)
```

   and update both call sites in the `except` block accordingly (`_publish_failure_artifacts(ctx=ctx, active=active, exc=exc)`). In the no-run-yet branch, `_start_failure_run`'s returned values are assigned onto `ctx` fields before publishing.
7. Convert the best-effort diagnostic — `_try_log_train_eval` gains a `ctx` parameter and records instead of swallowing:

```python
def _try_log_train_eval(
    *,
    ctx: TrialContext,
    run_id: str,
    active: Session,
    model,
    dataset_id: str,
    train_split: str,
    feature_registry,
) -> None:
    try:
        eval_dataset, _ = prepare_eval_dataset(
            session=active,
            dataset_id=dataset_id,
            split=train_split,
        )
        evaluate(
            session=active,
            model_run_id=run_id,
            eval_dataset_id=eval_dataset.id,
            label=train_split,
            set_as_primary_label=False,
            _model=model,
            _model_feature_registry=feature_registry,
        )
    except Exception as exc:  # noqa: BLE001 - diagnostic stays best-effort, but visibly so
        ctx.record_issue(exc, phase="evaluation", severity="warning")
```

   (update its call site to pass `ctx=ctx`).

- [ ] **Step 4: Run the full runner suite**

Run: `uv run pytest tests/unit/runner -q`
Expected: all PASS. `test_trial_folder_execution.py` exercises the whole `_run_trial` flow with fakes — failures here mean a missed mechanical replacement; fix until green.

- [ ] **Step 5: Commit**

```bash
git add automl/runner/trial.py tests/unit/runner/test_issues.py
git commit -m "refactor(runner): thread TrialContext through the trial; ledger published on both exit paths"
```

---

### Task 5: Record validation issues into the ledger

**Files:**
- Modify: `automl/runner/serving_validation.py` (accept the context; record timeout / signal / latency-not-measured)
- Modify: `automl/runner/trial.py` (pass `context=ctx` to `log_validation_artifacts`)
- Test: `tests/unit/runner/test_validation_errors.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/runner/test_validation_errors.py`:

```python
def test_timeout_records_ledger_issue(tmp_path, monkeypatch):
    from automl.runner.context import TrialContext

    def _raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["python"], timeout=7, stderr=b"slow")

    monkeypatch.setattr(serving_validation.subprocess, "run", _raise_timeout)
    ctx = TrialContext()
    serving_validation._run_pyfunc_validation(
        run_id="run123",
        active=_FakeSession(),
        input_parquet=tmp_path / "input.parquet",
        input_csv=tmp_path / "input.csv",
        expected_parquet=tmp_path / "expected.parquet",
        input_schema=tmp_path / "input_schema.json",
        report_path=tmp_path / "report.json",
        tolerance=1e-10,
        context=ctx,
    )
    (issue,) = ctx.issues.snapshot()
    assert issue["severity"] == "error"
    assert "exceeded" in issue["message"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/runner/test_validation_errors.py -q -k ledger`
Expected: FAIL — `TypeError: _run_pyfunc_validation() got an unexpected keyword argument 'context'`.

- [ ] **Step 3: Implement**

In `automl/runner/serving_validation.py`:

1. `log_validation_artifacts(...)` gains `context=None` (typed `"TrialContext | None"` via a `TYPE_CHECKING` import to avoid a load-time cycle if one appears; a plain untyped default is also fine) and passes it to `_run_pyfunc_validation(..., context=context)`.
2. `_run_pyfunc_validation(..., context=None)`:
   - timeout handler, after building `report`: `if context is not None: context.record_issue(report["error"], phase="validation", severity="error")`
   - signal guard (from plan 1), after building `report`: same call with the signal message.
3. The latency-not-measured branch in `log_validation_artifacts` (where `latency_detail` gets `"status": "not_measured"`): add

```python
            if context is not None:
                context.record_issue(
                    "validation latency not measured (validation failed)",
                    phase="validation_publish",
                    severity="warning",
                )
```

4. In `automl/runner/trial.py`, the `log_validation_artifacts(...)` call gains `context=ctx`.

- [ ] **Step 4: Run the runner suite + full unit suite + contracts**

Run: `uv run pytest tests/unit tests/contracts -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add automl/runner/serving_validation.py automl/runner/trial.py tests/unit/runner/test_validation_errors.py
git commit -m "feat(validation): timeout/signal/latency failures land in the trial issue ledger"
```

---

## Done criteria

- `uv run pytest tests/unit tests/contracts -q` green.
- `grep -n "except Exception" automl/runner/trial.py` shows no silent swallows — every handler either publishes failure artifacts or records a ledger issue.
- Every finished trial run in MLflow has `trial/issues.json` + a `trial.issue_count` tag (0 on a clean trial).
- A killed runner process leaves `issues.jsonl` in the trial dir with everything recorded up to the crash (boundary #3 — durable publication of that file is `docs/to-do/runner-crash-supervision.md`, not this plan).
