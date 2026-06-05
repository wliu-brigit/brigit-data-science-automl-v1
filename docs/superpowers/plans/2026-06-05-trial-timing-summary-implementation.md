# Trial Timing Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `timing/summary.json` the single canonical trial timing artifact, with flat high-level phases and consistent per-phase details in chronological order.

**Architecture:** Runner timing remains the source for detailed runner phases. A new `automl.trial.timing_summary` helper normalizes schema v2 and five-decimal rounding. Agent timeline publish enriches the existing trial artifact using hook spans, CLI step events, and matched MLflow run boundaries; if run boundaries are absent, coder non-runner time is derived from coder wall-clock minus runner duration.

**Tech Stack:** Python 3.12, MLflow run artifacts/metrics, existing agent timeline hook events, pytest.

---

## File Structure

- Create: `automl/trial/timing_summary.py`
  - Build runner-only v2 timing summaries.
  - Build enriched agent timing summaries with ordered `phases` and `phase_details`.
- Modify: `automl/trial/metadata.py`
  - Preserve `phase_details` when normalizing timing reports.
- Modify: `automl/mlflow/trial/artifacts/runner.py`
  - Log normalized v2 timing summaries.
- Modify: `automl/runner/timing_artifacts.py`
  - Log metrics from v2 summary while preserving existing runner detail metrics.
- Modify: `automl/runner/trial.py`
  - Pass the same normalized timing summary to `log_timing` and the manifest.
- Modify: `automl/agent/timeline/reconcile.py`
  - Keep proposer/coder start/end and matched run start/end on each iteration.
  - Keep CLI step `start_s` so setup and proposal handoff can be classified chronologically.
- Modify: `automl/agent/timeline/_publish.py`
  - Rewrite each trial `timing/summary.json` with enriched canonical timing when the runner artifact exists.

---

## Task 1: Timing Summary Helper

**Files:**
- Create: `automl/trial/timing_summary.py`
- Create: `tests/unit/trial/test_timing_summary.py`

- [ ] **Step 1: Add helper tests**
  - Assert `round_seconds()` returns five decimal places.
  - Assert `build_runner_timing_summary()` converts current runner snapshots into:
    - `schema_version: 2`
    - `phases: {"runner": total}`
    - `phase_details.runner.phases` preserving runner detail order.
  - Assert `enrich_agent_timing_summary()` emits phases in this order:
    `setup`, `proposer`, `proposal_handoff`, `coder_implementation`, `runner`, `coder_report`, `publish`.
  - Assert every phase has a `phase_details.<phase>.total_seconds`.

- [ ] **Step 2: Run failing tests**
  - Run: `uv run pytest tests/unit/trial/test_timing_summary.py -q`
  - Expected: FAIL because `automl.trial.timing_summary` does not exist.

- [ ] **Step 3: Implement helper**
  - Add rounding, runner summary build, step grouping, and enrichment functions.
  - Keep the public artifact clean: no `_internal`, no `started_at_s`, no `ended_at_s`.

- [ ] **Step 4: Run tests**
  - Run: `uv run pytest tests/unit/trial/test_timing_summary.py -q`
  - Expected: PASS.

## Task 2: Runner v2 Timing

**Files:**
- Modify: `automl/trial/metadata.py`
- Modify: `automl/mlflow/trial/artifacts/runner.py`
- Modify: `automl/runner/timing_artifacts.py`
- Modify: `automl/runner/trial.py`
- Modify: `tests/unit/mlflow/test_runner_artifacts.py`
- Modify: `tests/integration/runner/test_one_trial_local.py`

- [ ] **Step 1: Update runner artifact tests**
  - Expect `timing/summary.json` schema v2.
  - Expect runner details under `phase_details.runner.phases`.

- [ ] **Step 2: Run failing tests**
  - Run: `uv run pytest tests/unit/mlflow/test_runner_artifacts.py -q`
  - Expected: FAIL with old schema shape.

- [ ] **Step 3: Normalize runner timing writes**
  - Use `build_runner_timing_summary()` in runner artifact writes.
  - Keep metrics for `timing.total_seconds`, `timing.runner_seconds`, and each runner detail phase.
  - Keep legacy convenience metrics `time.fit_seconds`, `time.eval_seconds`, and `time.validation_seconds`.

- [ ] **Step 4: Run focused tests**
  - Run: `uv run pytest tests/unit/mlflow/test_runner_artifacts.py tests/integration/runner/test_one_trial_local.py -q`
  - Expected: PASS.

## Task 3: Agent Timing Enrichment

**Files:**
- Modify: `automl/agent/timeline/reconcile.py`
- Modify: `automl/agent/timeline/_publish.py`
- Modify: `tests/unit/agent/test_timeline.py`
- Modify: `tests/e2e/test_agent_timeline_hooks.py`

- [ ] **Step 1: Update publish tests**
  - Create or preserve a runner `timing/summary.json` artifact in the test run.
  - Assert publish rewrites `timing/summary.json` with high-level phases and per-phase details.
  - Assert setup/proposal handoff derive from CLI steps by time.

- [ ] **Step 2: Run failing tests**
  - Run: `uv run pytest tests/unit/agent/test_timeline.py::test_publish_backfills_trial_artifacts_when_real_hook_lacks_run_fields -q`
  - Expected: FAIL because publish does not enrich timing yet.

- [ ] **Step 3: Implement enrichment**
  - Carry spans and run boundaries in reconciliation.
  - Classify step events into setup and proposal handoff.
  - Load existing `timing/summary.json`; if missing, leave timing unchanged.
  - Log enriched `timing/summary.json` to the same path.

- [ ] **Step 4: Run focused timing tests**
  - Run: `uv run pytest tests/unit/trial/test_timing_summary.py tests/unit/mlflow/test_runner_artifacts.py tests/unit/agent/test_timeline.py -q`
  - Expected: PASS.

## Task 4: Verification

**Files:**
- No additional files.

- [ ] **Step 1: Run focused suite**
  - Run: `uv run pytest tests/unit/trial/test_timing_summary.py tests/unit/mlflow/test_runner_artifacts.py tests/unit/agent/test_timeline.py tests/integration/runner/test_one_trial_local.py -q`
  - Expected: PASS.

- [ ] **Step 2: Run broader affected tests**
  - Run: `uv run pytest tests/unit tests/contracts tests/integration -q`
  - Expected: PASS.
