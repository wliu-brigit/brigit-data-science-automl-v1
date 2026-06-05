# Dataset Active Pointer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace implicit latest-dataset fallback with one explicit active dataset pointer shared by data, proposer, runner, profile, CLI, and notebooks.

**Architecture:** Dataset records remain `datasets/<dataset_id>/dataset.json`. Activation writes the MLflow experiment tag `data.active_dataset_id` plus the human-readable mirror artifact `datasets/active_pointer.json`. Default data reads resolve and validate that pointer; explicit dataset-id APIs remain the bypass.

**Tech Stack:** Python 3.12, MLflow experiment tags/artifacts, existing `automl.data` and `automl.cli` modules, pytest.

---

## File Structure

- Create: `automl/data/selection.py`
  - Owns `activate_dataset(...)`, `resolve_active_dataset_id(...)`, and `resolve_active_dataset(...)`.
  - Validates tag/artifact consistency and dataset-record existence.
- Modify: `automl/mlflow/experiment/artifacts.py`
  - Adds `write_active_dataset_pointer(...)` and `read_active_dataset_pointer(...)`.
- Modify: `automl/data/__init__.py`
  - Re-exports selection helpers.
- Modify: `automl/data/pipeline.py`
  - First materialize and refresh materialize activate the resulting dataset through the core API.
  - Non-refresh materialize attaches to active only; if records exist but pointer is missing/broken, it errors.
- Modify: `automl/data/registry.py`
  - `load_dataset(...)` resolves active only; no latest fallback.
- Modify: `automl/data/profile.py`
  - Default `profile()` and `get_profile()` resolve active only.
- Modify: `automl/agent/proposer_context.py`
  - Proposer context reads the same active dataset as runner/profile.
- Modify: `automl/cli/data.py`
  - Adds `automl data activate <dataset_id>`.
- Modify: `automl/cli/trial.py`, `automl/cli/_trial_actions.py`, `automl/runner/trial.py`
  - Adds `automl trial run --dataset-id <dataset_id>` as a trial-only override.

---

## Task 1: Active Pointer Artifact Helpers

**Files:**
- Modify: `automl/mlflow/experiment/artifacts.py`
- Modify: `tests/unit/mlflow/test_experiment_dataset_artifacts.py`

- [ ] **Step 1: Add tests**
  - Assert `write_active_dataset_pointer("v2_good")` writes `datasets/active_pointer.json`.
  - Assert `read_active_dataset_pointer()` returns `{"schema_version": 1, "active_dataset_id": "v2_good"}`.
  - Assert absent pointer returns `None`.

- [ ] **Step 2: Run failing tests**
  - Run: `uv run pytest tests/unit/mlflow/test_experiment_dataset_artifacts.py -q`
  - Expected: FAIL because helpers do not exist.

- [ ] **Step 3: Implement helpers**
  - Use the experiment overview run.
  - Keep pointer payload minimal and human-readable.

## Task 2: Core Selection API

**Files:**
- Create: `automl/data/selection.py`
- Modify: `automl/data/__init__.py`
- Create: `tests/unit/data/test_dataset_selection.py`

- [ ] **Step 1: Add tests**
  - `activate_dataset()` rejects a missing dataset record.
  - `activate_dataset()` writes both tag and pointer artifact.
  - `resolve_active_dataset()` rejects missing tag, missing artifact, tag/artifact mismatch, and missing pointed record.
  - `resolve_active_dataset()` returns the pointed dataset when tag/artifact/record agree.

- [ ] **Step 2: Run failing tests**
  - Run: `uv run pytest tests/unit/data/test_dataset_selection.py -q`
  - Expected: FAIL because `automl.data.selection` does not exist.

- [ ] **Step 3: Implement API**
  - Bind to the active experiment.
  - Convert pointer storage inconsistencies to `DataError`.
  - Keep explicit missing dataset ids as `KeyError`.

## Task 3: Read and Write Paths

**Files:**
- Modify: `automl/data/pipeline.py`
- Modify: `automl/data/registry.py`
- Modify: `automl/data/profile.py`
- Modify: `automl/agent/proposer_context.py`
- Modify: `tests/integration/data_pipeline/test_materialize_load.py`
- Modify: `tests/integration/data_pipeline/test_profile_integration.py`
- Modify: `tests/unit/agent/test_proposer_context.py`

- [ ] **Step 1: Add tests**
  - First materialize writes both active tag and `datasets/active_pointer.json`.
  - Refresh materialize activates the resulting dataset.
  - Existing records with no active pointer make default load/profile fail.
  - Proposer context uses `resolve_active_dataset()` rather than active-or-latest.

- [ ] **Step 2: Implement path changes**
  - Replace direct `set_active_dataset()` calls with `activate_dataset()`.
  - Replace active-or-latest reads with `resolve_active_dataset()`.
  - Keep `load_dataset_by_id()` unchanged.

## Task 4: CLI and Trial Override

**Files:**
- Modify: `automl/cli/data.py`
- Modify: `automl/cli/trial.py`
- Modify: `automl/cli/_trial_actions.py`
- Modify: `automl/runner/trial.py`
- Modify: `tests/unit/cli/test_cli_catalog.py`
- Modify: `tests/integration/runner/test_one_trial_local.py`

- [ ] **Step 1: Add tests**
  - `automl data activate <dataset_id>` calls the core activation API and prints the activated dataset.
  - `automl trial run --dataset-id <dataset_id>` passes the override to `run_trial()`.
  - Runner loads the override via `load_dataset_by_id()` and records the override dataset in the existing trial data contract.

- [ ] **Step 2: Implement CLI and runner override**
  - The override applies only to that trial and does not mutate the active pointer.

## Task 5: Verification

**Files:**
- No additional files.

- [ ] **Step 1: Run focused suite**
  - Run: `uv run pytest tests/unit/data/test_dataset_selection.py tests/unit/mlflow/test_experiment_dataset_artifacts.py tests/unit/cli/test_cli_catalog.py tests/unit/agent/test_proposer_context.py tests/integration/data_pipeline/test_materialize_load.py tests/integration/runner/test_one_trial_local.py -q`
  - Expected: PASS.

- [ ] **Step 2: Run broader affected tests**
  - Run: `uv run pytest tests/unit tests/contracts tests/integration -q`
  - Expected: PASS.
