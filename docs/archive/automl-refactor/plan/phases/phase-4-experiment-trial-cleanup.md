# Phase 4 Experiment Trial Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development`
> (recommended) or `superpowers:executing-plans` to implement this plan task-by-task.
> Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** pass A4.1-A4.4 by adding the thin experiment/trial read models and the cleanup
cascade needed to inspect, compare, summarize, and delete one experiment safely.

**Architecture:** Phase 4 is split into two task groups after a shared seam-read foundation.
Trial/experiment reads are typed, cheap-by-default domain views composed over the MLflow seam;
cleanup is a project-owned cascade engine with thin experiment/trial wrappers and preview-first
delete semantics. The runner, eval persistence, and Phase 3 external-eval path stay unchanged
except where read models need to consume their existing tags/artifacts.

**Tech Stack:** Python 3.11 via `uv`; pytest; file-backed MLflow for unit/integration tests;
fake GCS clients for destructive cleanup tests; original local MLflow
`http://127.0.0.1:54321` with this worktree's `.env` loaded for the opt-in Phase 4 gate; GCS
bucket `gs://automl-homecredit-kaggle-wliu`.

**Acceptance:** `plan/acceptance-checklist.md` rows **A4.1-A4.4**.

**Completion evidence (2026-05-28):**

```text
uv run pytest tests/unit tests/contracts tests/integration -v -> 222 passed, 2 warnings
AUTOML_PHASE4_E2E=1 MLFLOW_TRACKING_URI=http://127.0.0.1:54321 uv run pytest tests/e2e/test_phase4_experiment_trial_cleanup.py -v -> 1 passed, 26 warnings
AUTOML_PHASE3_E2E=1 MLFLOW_TRACKING_URI=http://127.0.0.1:54321 uv run pytest tests/e2e/test_phase3_eval_breadth.py -v -> 1 passed, 19 warnings
uv run pytest tests/contracts -v -> 9 passed
rg 'automl_legacy' automl projects tests -> only contract/doc text hits
rg '(^|\s)(import|from) mlflow' automl projects tests -> only automl/mlflow seam hits
```

Post-implementation review hardening fixed four Phase 4 edge cases: `LeaderboardData.n_unscored`
now counts trials missing the resolved metric rather than display-limit overflow; trial cleanup
rebinds explicit sessions before parent-run lookup; trial hard-delete runs `mlflow gc` with
`--run-ids` instead of an unfiltered GC command; and cleanup GCS prefixes handle empty
`ProjectConfig.gcs_prefix` without introducing a double slash. Project-scope cleanup also now
rejects a name that does not match the active session project.

---

## Review Decisions

This plan is grounded in specs 03/09/10, contracts from 00/01/02/07/08, the Phase 3 code, and
legacy read/cleanup flows. The first review resolved these Phase 4 decisions:

- **A4 uses CLI-shaped wording, but Phase 6 owns the full CLI catalog.** Phase 4 implements
  domain/library APIs with names and signatures that the eventual CLI can wrap directly. Do not
  pull the CLI dispatcher forward unless a Phase 4 gate proves literal command invocation is
  necessary.
- **Leaderboard metric selection is user/config-defined.** Default to
  `session.config.primary_metric` exactly, and accept an explicit `metric=` override for callers
  who want another logged key such as `test.auc`. Do not infer the ranking metric by scanning
  MLflow runs. Report the resolved key in `LeaderboardData.metric`.
- **Cleanup namespace safety must not rely on `ParentExperimentRef` alone.** Spec 10's
  `ParentExperimentRef` omits a `namespace` field, while A4 requires cleanup never cross
  namespace. `mlflow.trial.get_parent_experiment()` parses using the current bound namespace, and
  cleanup additionally compares the full parent `mlflow_experiment_name` to the exact route
  expected for the active session before deleting.
- **Cleanup e2e uses a dedicated QA namespace and unique experiment id.** Unit/integration tests
  prove preview, idempotency, and hard-delete subprocess behavior with fakes/patches. The external
  A4 gate may soft-delete real test-owned blobs only under a unique `qa-phase4-*` namespace. Do
  not run hard GC against the shared harness unless explicitly approved.

## Evidence Read

- `spec/09-experiment.md`: `ExperimentOverview` is the Experiment noun; raw searches live in the
  MLflow seam; experiment views compose `TrialSummary`/`TrialDetails`; `LeaderboardData` and
  `ComparisonResult` are typed; `summary` stays a dict; `recent_failures` and
  `strategies_attempted` are in scope.
- `spec/10-trial.md`: `TrialSummary` is the cheap row; `TrialDetails` is the deep single-run read;
  `show_trial()` enriches `get_details()` with `EvalResult` artifacts; public `load_model()` lives
  in `trial/show.py`; create/fork/promote/package_model are trial-domain work but not required by
  A4.
- `spec/03-cleanup.md`: cleanup is preview-by-default, project-owned, and ordered
  MLflow -> GCS -> local. One invocation cleans one `(namespace, dry_run)` universe. `--apply` is
  the destructive opt-in; soft delete is default; `hard_delete=True` runs `mlflow gc`; reruns are
  idempotent.
- `spec/02-mlflow-seam.md`: domains do not import PyPI `mlflow`; seam reads return typed domain
  objects. `mlflow.experiment.list_trials` and `top_n_by_metric` return `TrialSummary`;
  `mlflow.trial.get_details`, `list_artifacts`, `get_parent_experiment`, and eval/model artifact
  loaders are the read primitives Phase 4 needs.
- Current code: `automl/experiment/` and `automl/trial/` are empty; `mlflow.experiment.queries`
  returns raw MLflow run objects; `mlflow.project.list_experiments()` is a stub returning
  `[]`; Phase 3 already has eval/result/predictions artifact loaders and seam-local
  `artifacts.load_model`.
- Legacy source: `automl_legacy/inspect/views.py` has the old leaderboard/compare/show/model-load
  shapes; `automl_legacy/loop_context/queries.py` and `summary.py` have the aggregation logic;
  `automl_legacy/cleanup.py` has the old cascade ordering and MLflow GC command; the new spec
  intentionally changes cleanup confirmation and per-blob error handling.

## Planning Interpretations

- Phase 4 covers only the A4 behavior gate. Leave agent proposer context, full CLI catalog,
  broad namespace/dry-run surface, proposal validation, hooks, trial create/fork/promote,
  notebook cleanup, and cutover out of scope.
- `trial.types.TrialStatus` lands as the read-model status enum. Keep `runner.TrialStatus`
  unchanged in Phase 4 to avoid a runner refactor; seam readers map status by string value.
  Revisit unifying the enum only if a Phase 4 test proves duplication is harmful.
- `TrialSummary.training_origin` and `hypothesis` default to empty strings when older Phase 1-3
  runs lack those tags. That is an additive read-model behavior, not a runner rewrite.
- `TrialDetails.evaluations is None` means the cheap seam read did not load eval artifacts;
  `[]` means `trial.show_trial()` loaded and found none.
- Cleanup tests must prove no mutation before `apply=True` and must use fake GCS or unique
  test-owned prefixes before any real delete is exercised.
- Example notebooks still referencing `eval_snapshot` remain deferred unless A4 code touches
  notebook/user surfaces.

## File Structure

Create read-model domain files:

- `automl/trial/types.py`: `TrialStatus`, `ArtifactRef`, `TrialSummary`, `TrialDetails`,
  `ParentExperimentRef`, and forward-compatible `from_dict` loaders.
- `automl/trial/show.py`: `show_trial(run_id, *, session=None)` and
  `load_model(run_id, *, session=None)`.
- `automl/experiment/store.py`: `ExperimentOverview` and `Experiment = ExperimentOverview`.
- `automl/experiment/lifecycle.py`: `create(experiment_id=None, *, session=None)`.
- `automl/experiment/views/types.py`: `LeaderboardData`, `MetricDelta`, `ComparisonResult`.
- `automl/experiment/views/leaderboard.py`: `leaderboard(...) -> LeaderboardData`.
- `automl/experiment/views/compare.py`: `compare(run_ids, *, session=None)`.
- `automl/experiment/views/queries.py`: `recent_failures`, `strategies_attempted`.
- `automl/experiment/views/summary.py`: `load_mlflow_context`, `build_summary_from_context`,
  `build_summary`, `experiments`.
- `automl/experiment/cleanup.py`: thin `delete(experiment_id, *, apply, hard_delete, session)`.
- `automl/project/cleanup.py`: shared cleanup dataclasses, plan builder, apply engine.
- `automl/trial/cleanup.py`: thin trial delete wrapper.

Modify seam and support files:

- `automl/mlflow/experiment/queries.py`: return `TrialSummary`, add training-origin filtering,
  count/search helpers, and shared raw-run builders.
- `automl/mlflow/trial/reads.py`: new read primitives for `TrialDetails`, metrics, artifacts,
  and parent experiment parsing.
- `automl/mlflow/trial/__init__.py`: export read primitives.
- `automl/mlflow/project/overview.py`: implement `list_experiments()` for logical experiment ids.
- `automl/mlflow/experiment/lifecycle.py`: make overview functions return `ExperimentOverview`.
- `automl/utils/io/gcs.py`: add delete helpers used by cleanup.
- `automl/experiment/__init__.py`, `automl/experiment/views/__init__.py`, `automl/trial/__init__.py`,
  and `automl/__init__.py`: expose only Phase 4 public types/functions. Leave project cleanup
  importable from `automl.project.cleanup` unless a later phase promotes it.

Create/modify tests:

- `tests/unit/trial/test_types.py`
- `tests/unit/trial/test_show.py`
- `tests/unit/experiment/test_view_types.py`
- `tests/unit/experiment/test_views.py`
- `tests/unit/project/test_cleanup.py`
- `tests/unit/mlflow/test_trial_reads.py`
- `tests/unit/mlflow/test_experiment_read_queries.py`
- `tests/integration/cleanup/test_experiment_delete.py`
- `tests/e2e/test_phase4_experiment_trial_cleanup.py`

## Task DAG

```
P4.0 baseline guard
  -> P4.1 trial read types and MLflow seam run builders
  -> P4.2 trial show/load_model domain read API
  -> P4.3 experiment overview and project experiment listing seam
  -> P4.4 experiment view types, leaderboard, compare, summary, query helpers
  -> P4.5 cleanup dataclasses and dry-run plan builder
  -> P4.6 cleanup apply engine, GCS delete helpers, and hard-delete command
  -> P4.7 experiment/trial cleanup wrappers and local integration safety tests
  -> P4.8 Phase 4 e2e gate plus Phase 3 regression gate
  -> P4.9 docs closeout and commit
```

Each task is a review boundary. If specs contradict running evidence, stop and flag it before
changing code.

---

## P4.0 - Baseline Guard And Scope Ratchet

**Files:**
- Modify: no production files.
- Test/read: existing Phase 3 tests and contract ratchets.

**Specs:** `plan/implementation-strategy.md` sections 5-6; `plan/acceptance-checklist.md` A3/A4.

**Steps:**
- [x] Run the Phase 3 non-external suite before changing code.

  Run:
  ```bash
  uv run pytest tests/unit tests/contracts tests/integration -v
  ```
  Expected: PASS. If this fails, fix the existing regression before starting Phase 4.

- [x] Run architecture ratchets.

  Run:
  ```bash
  uv run pytest tests/contracts -v
  rg 'automl_legacy' automl projects tests
  rg '(^|\s)(import|from) mlflow' automl projects tests
  ```
  Expected: contract tests pass; `automl_legacy` appears only in ratchet/doc text; PyPI
  `mlflow` imports appear only under `automl/mlflow/` or tests intentionally verifying the seam.

- [x] Confirm Phase 4 is library-domain scoped.

  Check:
  ```bash
  wc -l automl/cli/__init__.py
  ```
  Expected: `0 automl/cli/__init__.py`. Do not add CLI dispatcher code in Phase 4 unless this
  plan is explicitly amended by a later gate failure.

**Acceptance:** Starting line is green and Phase 4 does not accidentally become Phase 6.

---

## P4.1 - Trial Read Types And MLflow Seam Run Builders

**Files:**
- Create: `automl/trial/types.py`
- Modify: `automl/trial/__init__.py`
- Create: `automl/mlflow/trial/reads.py`
- Modify: `automl/mlflow/trial/__init__.py`
- Modify: `automl/mlflow/experiment/queries.py`
- Test: `tests/unit/trial/test_types.py`
- Test: `tests/unit/mlflow/test_trial_reads.py`
- Test: `tests/unit/mlflow/test_experiment_read_queries.py`

**Specs:** `spec/10-trial.md` Q1-Q4, Q10, Q13-Q14; `spec/02-mlflow-seam.md` sections
6.2.2, 6.3.3, 8, and 11; `spec/09-experiment.md` Q3 and Q6.

**Legacy source:**
- `automl_legacy/mlflow/store.py::_run_to_trial_summary`
- `automl_legacy/loop_context/queries.py::show_trial`

**Migration rows:** `mlflow/store.py::get_trial_summaries`; `loop_context/queries.py::top_n_by_metric`;
`loop_context/queries.py::show_trial` drop path; new `TrialSummary`/`TrialDetails` homes in
`trial/types.py`.

**Steps:**
- [x] Write failing type round-trip tests.

  Add `tests/unit/trial/test_types.py` with tests equivalent to:
  ```python
  from automl.eval import EvalResult
  from automl.trial.types import ArtifactRef, ParentExperimentRef
  from automl.trial.types import TrialDetails, TrialStatus, TrialSummary


  def test_trial_summary_from_dict_normalizes_status_and_unknown_fields():
      summary = TrialSummary.from_dict(
          {
              "schema_version": 1,
              "run_id": "run-1",
              "slug": "baseline",
              "strategy": "logistic",
              "status": "FINISHED",
              "primary_metric_name": "auc",
              "primary_metric_value": 0.71,
              "trial_number": "3",
              "hypothesis": "numeric baseline",
              "training_origin": "automl",
              "training_time_s": "12.5",
              "n_features": "42",
              "ignored_newer_field": "ok",
          }
      )

      assert summary.status is TrialStatus.FINISHED
      assert summary.trial_number == 3
      assert summary.training_time_s == 12.5
      assert summary.n_features == 42


  def test_trial_details_from_dict_loads_nested_artifacts_and_eval_results():
      details = TrialDetails.from_dict(
          {
              "run_id": "run-1",
              "status": "FAILED",
              "params": {"C": "1.0"},
              "metrics": {"test.auc": 0.4},
              "tags": {"automl.trial.strategy": "tree"},
              "artifacts": [{"path": "eval/test/results.json", "file_size": 123}],
              "evaluations": [
                  {
                      "label": "test",
                      "eval_dataset_id": "eval_abc",
                      "eval_dataset_kind": "split_view",
                      "predictions_uri": "",
                      "predictions_manifest_uri": "",
                      "augmentations_used": [],
                      "primary": "auc",
                      "metrics": [{"name": "auc", "value": 0.4}],
                      "computed_at": "2026-05-28T00:00:00Z",
                  }
              ],
          }
      )

      assert details.status is TrialStatus.FAILED
      assert details.artifacts == (ArtifactRef(path="eval/test/results.json", file_size=123),)
      assert isinstance(details.evaluations[0], EvalResult)


  def test_trial_details_none_vs_loaded_empty_evaluations():
      cheap = TrialDetails(run_id="run-1", evaluations=None)
      loaded_empty = TrialDetails(run_id="run-1", evaluations=())

      assert cheap.evaluations is None
      assert loaded_empty.evaluations == ()


  def test_parent_experiment_ref_round_trips_full_name():
      parent = ParentExperimentRef.from_dict(
          {
              "mlflow_experiment_id": "42",
              "mlflow_experiment_name": "qa/dry_run/home_credit/baseline",
              "dry_run": True,
              "project_name": "home_credit",
              "experiment_id": "baseline",
          }
      )

      assert parent.mlflow_experiment_id == "42"
      assert parent.dry_run is True
  ```

  Run:
  ```bash
  uv run pytest tests/unit/trial/test_types.py -v
  ```
  Expected: FAIL because `automl.trial.types` does not exist.

- [x] Implement `automl/trial/types.py`.

  Required behavior:
  - `TrialStatus(str, Enum)` with `UNKNOWN`, `RUNNING`, `FINISHED`, `FAILED`, `KILLED`.
  - `ArtifactRef(path="", file_size=None)` as a schema-less nested type used consistently in
    `TrialDetails.artifacts`.
  - `TrialSummary` fields from spec 10 Q3, with `from_dict` stripping unknown keys and coercing
    numeric strings for `trial_number`, `training_time_s`, and `n_features`.
  - `TrialDetails` fields from spec 10 Q1/Q4, with `evaluations: tuple[EvalResult, ...] | None`.
  - `ParentExperimentRef` fields from spec 03/10, including the full
    `mlflow_experiment_name` string so cleanup can do exact route checks.
  - Private helpers only for local coercion; no PyPI `mlflow` imports.

- [x] Write failing MLflow read tests.

  Add `tests/unit/mlflow/test_trial_reads.py` and `tests/unit/mlflow/test_experiment_read_queries.py`
  with tests equivalent to:
  ```python
  import pytest

  from automl.mlflow import client, experiment, tags, trial
  from automl.trial.types import ParentExperimentRef, TrialDetails, TrialStatus, TrialSummary


  @pytest.fixture
  def bound_file_mlflow(tmp_path):
      client.clear()
      client.bind(
          tracking_uri=(tmp_path / "mlruns").as_uri(),
          bucket="",
          gcs_prefix="automl-root",
          project_name="home_credit",
          experiment_id="baseline",
          namespace="qa",
      )
      yield
      client.clear()


  def test_list_trials_returns_typed_summaries_newest_first(bound_file_mlflow):
      experiment.ensure()
      with trial.active(slug="unscored", strategy="baseline") as unscored_id:
          trial.set_tags(unscored_id, {tags.TRIAL_NUMBER: "1", tags.TRIAL_ID: "1_unscored"})
      with trial.active(slug="scored", strategy="tree") as scored_id:
          trial.set_tags(scored_id, {tags.TRIAL_NUMBER: "2", tags.TRIAL_ID: "2_scored"})
          trial.log_metric(scored_id, "test.auc", 0.81)
          trial.log_metric(scored_id, "auc", 0.81)
          trial.set_tag(scored_id, tags.EVAL_PRIMARY_METRIC, "auc")

      rows = experiment.list_trials()

      assert [row.run_id for row in rows] == [scored_id, unscored_id]
      assert all(isinstance(row, TrialSummary) for row in rows)
      assert rows[0].status is TrialStatus.FINISHED
      assert rows[0].primary_metric_name == "auc"
      assert rows[0].primary_metric_value == 0.81


  def test_top_n_by_metric_filters_training_origin_and_returns_typed_rows(bound_file_mlflow):
      experiment.ensure()
      with trial.active(slug="automl", strategy="baseline") as run_a:
          trial.log_metric(run_a, "test.auc", 0.7)
          trial.set_tag(run_a, "automl.trial.training_origin", "automl")
      with trial.active(slug="human", strategy="baseline") as run_b:
          trial.log_metric(run_b, "test.auc", 0.9)
          trial.set_tag(run_b, "automl.trial.training_origin", "human")

      rows = experiment.top_n_by_metric("test.auc", n=5, training_origin="automl")

      assert [row.run_id for row in rows] == [run_a]


  def test_get_details_and_parent_experiment_return_typed_values(bound_file_mlflow):
      experiment.ensure()
      with trial.active(slug="baseline", strategy="logistic") as run_id:
          trial.log_metric(run_id, "test.auc", 0.72)
          trial.log_param(run_id, "solver", "liblinear")

      details = trial.get_details(run_id)
      parent = trial.get_parent_experiment(run_id)

      assert isinstance(details, TrialDetails)
      assert details.run_id == run_id
      assert details.metrics["test.auc"] == 0.72
      assert details.params["solver"] == "liblinear"
      assert details.evaluations is None
      assert isinstance(parent, ParentExperimentRef)
      assert parent.mlflow_experiment_name == "qa/home_credit/baseline"
      assert parent.project_name == "home_credit"
      assert parent.experiment_id == "baseline"
  ```

  Run:
  ```bash
  uv run pytest tests/unit/mlflow/test_trial_reads.py tests/unit/mlflow/test_experiment_read_queries.py -v
  ```
  Expected: FAIL because the read modules and typed builders do not exist.

- [x] Implement `automl/mlflow/trial/reads.py`.

  Required behavior:
  - `get_details(run_id) -> TrialDetails` wraps `client.raw().get_run(run_id)`.
  - `get_metrics(run_id) -> dict[str, float]` returns latest run metrics.
  - `list_artifacts(run_id) -> tuple[ArtifactRef, ...]` recursively walks MLflow artifacts.
  - `get_parent_experiment(run_id) -> ParentExperimentRef` resolves the run's MLflow experiment,
    parses the route using the active `client.bound().namespace`, and preserves the full route name.
  - Backend exceptions wrap in `StorageError`.

- [x] Modify `automl/mlflow/experiment/queries.py` to return typed `TrialSummary`.

  Required behavior:
  - Keep `next_trial_number` behavior.
  - `list_trials(..., status=None, training_origin=None)` returns newest-first
    `list[TrialSummary]`.
  - `top_n_by_metric(metric, n, ascending=False, experiment_id=None, training_origin=None)`
    returns only rows that computed `metric`.
  - `search_trials(filter_string, ...)` remains a mid-level seam escape hatch but returns
    typed rows, not raw run objects.
  - Shared private builder maps Phase 3 tags:
    `automl.trial.slug`, `automl.trial.strategy`, `automl.trial.status`,
    `automl.trial.id`, `automl.trial.number`, `automl.trial.parent_run_id`,
    `automl.trial.dataset_hash` or `data.identity_hash`, and
    `automl.trial.eval.primary_metric`.
  - If `tags.EVAL_PRIMARY_METRIC` is set, read the bare metric value from run metrics for
    `primary_metric_value`; otherwise use `None`.

- [x] Export read primitives from `automl/mlflow/trial/__init__.py` and trial types from
  `automl/trial/__init__.py`.

- [x] Run targeted tests.

  Run:
  ```bash
  uv run pytest tests/unit/trial/test_types.py tests/unit/mlflow/test_trial_reads.py tests/unit/mlflow/test_experiment_read_queries.py -v
  ```
  Expected: PASS.

**Acceptance:** seam read primitives return typed trial objects and no domain outside
`automl/mlflow/**` imports PyPI `mlflow`.

---

## P4.2 - Trial Show And Public Load Model

**Files:**
- Create: `automl/trial/show.py`
- Modify: `automl/trial/__init__.py`
- Test: `tests/unit/trial/test_show.py`
- Test: `tests/unit/mlflow/test_eval_predictions_artifacts.py`

**Specs:** `spec/10-trial.md` Q1, Q4, Q8, Q10; `spec/07-eval.md` result type; `spec/02-mlflow-seam.md`
section 10.

**Legacy source:**
- `automl_legacy/inspect/views.py::show_trial`
- `automl_legacy/inspect/views.py::load_model`

**Migration rows:** `inspect/views.py::show_trial`; `inspect/views.py::load_model`.

**Steps:**
- [x] Write failing tests for `trial.show_trial()` and `trial.load_model()`.

  Add `tests/unit/trial/test_show.py` with tests equivalent to:
  ```python
  import pytest

  from automl.eval import EvalResult
  from automl.mlflow import client, experiment, trial
  from automl.mlflow.trial import artifacts
  from automl.trial import load_model, show_trial


  @pytest.fixture
  def bound_file_mlflow(tmp_path):
      client.clear()
      client.bind(
          tracking_uri=(tmp_path / "mlruns").as_uri(),
          bucket="",
          gcs_prefix="automl-root",
          project_name="home_credit",
          experiment_id="baseline",
      )
      yield
      client.clear()


  def test_show_trial_enriches_get_details_with_eval_results(bound_file_mlflow):
      experiment.ensure()
      with trial.active(slug="baseline", strategy="logistic") as run_id:
          result = EvalResult(
              label="test",
              eval_dataset_id="eval_123",
              eval_dataset_kind="split_view",
              predictions_uri="",
              predictions_manifest_uri="",
              augmentations_used=(),
              primary="auc",
              metrics=({"name": "auc", "value": 0.8},),
              computed_at="2026-05-28T00:00:00Z",
          )
          artifacts.write_eval(run_id, "test", result)

      details = show_trial(run_id)

      assert details.run_id == run_id
      assert details.evaluations == (result,)


  def test_show_trial_returns_loaded_empty_evaluations_when_none_exist(bound_file_mlflow):
      experiment.ensure()
      with trial.active(slug="baseline", strategy="logistic") as run_id:
          pass

      assert show_trial(run_id).evaluations == ()


  def test_load_model_delegates_to_packaged_model_artifact(bound_file_mlflow):
      experiment.ensure()
      payload = {"model": "round-trip"}
      with trial.active(slug="baseline", strategy="logistic") as run_id:
          artifacts.write_model(run_id, payload)

      assert load_model(run_id) == payload
  ```

  Run:
  ```bash
  uv run pytest tests/unit/trial/test_show.py -v
  ```
  Expected: FAIL because `trial.show` does not exist.

- [x] Implement `automl/trial/show.py`.

  Required behavior:
  - Resolve `session` with the standard explicit-or-active idiom so calls are consistent with
    Tier 2 conventions. The session is mainly for ensuring the seam is bound.
  - `show_trial(run_id, *, session=None)` calls `automl.mlflow.trial.get_details(run_id)`, loads
    `automl.mlflow.trial.artifacts.list_eval(run_id)`, then loads each label via `load_eval`.
  - Return a new `TrialDetails` with populated `evaluations`; use `()` when no evals exist.
  - `load_model(run_id, *, session=None)` calls `automl.mlflow.trial.artifacts.load_model(run_id)`.

- [x] Export `show_trial`, `load_model`, and read types from `automl/trial/__init__.py`.

- [x] Run targeted tests.

  Run:
  ```bash
  uv run pytest tests/unit/trial/test_show.py tests/unit/mlflow/test_eval_predictions_artifacts.py -v
  ```
  Expected: PASS.

**Acceptance:** A4.2 has its trial-domain read API and model round-trip foundation.

---

## P4.3 - Experiment Overview And Project Experiment Listing Seam

**Files:**
- Create: `automl/experiment/store.py`
- Create: `automl/experiment/lifecycle.py`
- Modify: `automl/experiment/__init__.py`
- Modify: `automl/mlflow/experiment/lifecycle.py`
- Modify: `automl/mlflow/project/overview.py`
- Test: `tests/unit/experiment/test_overview_lifecycle.py`
- Test: `tests/unit/mlflow/test_experiment_read_queries.py`

**Specs:** `spec/09-experiment.md` Q1-Q2, Q6, Q12; `spec/02-mlflow-seam.md` sections 6.1
and 6.2.1.

**Legacy source:**
- `automl_legacy/mlflow/store.py::ensure_experiment_overview`
- `automl_legacy/inspect/views.py::experiments`

**Migration rows:** `mlflow/store.py::ensure_experiment_overview`; `inspect/views.py::experiments`.

**Steps:**
- [x] Write failing tests for `ExperimentOverview`, `create()`, and logical experiment listing.

  Add `tests/unit/experiment/test_overview_lifecycle.py` with tests equivalent to:
  ```python
  from automl.experiment import Experiment, ExperimentOverview, create
  from automl.mlflow import client, project
  from automl.project import ProjectConfig, Session


  def test_experiment_overview_from_dict_strips_unknown_fields():
      overview = ExperimentOverview.from_dict(
          {
              "experiment_id": "baseline",
              "project_name": "home_credit",
              "created_at": "2026-05-28T00:00:00Z",
              "dry_run": True,
              "future": "ignored",
          }
      )

      assert overview.experiment_id == "baseline"
      assert overview.dry_run is True
      assert Experiment is ExperimentOverview


  def test_create_returns_experiment_overview(tmp_path):
      client.clear()
      config = ProjectConfig(
          project_name="home_credit",
          repo_root=tmp_path,
          project_dir=tmp_path / "projects" / "home_credit",
          gcs_prefix="automl-root",
          mlflow_tracking_uri=(tmp_path / "mlruns").as_uri(),
      )
      active = Session(config=config, experiment_id="baseline", dry_run=True)
      client.bind(
          tracking_uri=config.mlflow_tracking_uri,
          bucket="",
          gcs_prefix=config.gcs_prefix,
          project_name=config.project_name,
          experiment_id="baseline",
          dry_run=True,
      )

      overview = create(session=active)

      assert overview.experiment_id == "baseline"
      assert overview.project_name == "home_credit"
      assert overview.dry_run is True
      assert overview.created_at


  def test_project_list_experiments_returns_logical_active_ids(tmp_path):
      client.clear()
      client.bind(
          tracking_uri=(tmp_path / "mlruns").as_uri(),
          bucket="",
          gcs_prefix="automl-root",
          project_name="home_credit",
          experiment_id="baseline",
      )
      create(experiment_id="baseline")
      create(experiment_id="second")
      # Create routes that must be excluded.
      mlflow_client = client.raw()
      mlflow_client.create_experiment("home_credit/overview")
      mlflow_client.create_experiment("home_credit/baseline/nested")
      mlflow_client.create_experiment("other_project/baseline")

      assert project.list_experiments() == ["baseline", "second"]
  ```

  Run:
  ```bash
  uv run pytest tests/unit/experiment/test_overview_lifecycle.py -v
  ```
  Expected: FAIL because experiment lifecycle/store are not implemented and project listing is empty.

- [x] Implement `automl/experiment/store.py`.

  Required behavior:
  - Frozen `ExperimentOverview` with fields from spec 09/02 and `from_dict`.
  - `Experiment = ExperimentOverview`.
  - No PyPI `mlflow` imports.

- [x] Implement `automl/experiment/lifecycle.py`.

  Required behavior:
  - `create(experiment_id=None, *, session=None)` resolves session, calls seam
    `mlflow.experiment.ensure()` and `ensure_overview()`, and returns `ExperimentOverview`.

- [x] Update `automl/mlflow/experiment/lifecycle.py`.

  Required behavior:
  - `ensure_overview()` creates or returns an overview run tagged as `automl.run_kind =
    "experiment_overview"`, `automl.experiment_id`, `automl.project_name`, `automl.dry_run`,
    and `automl.created_at`.
  - `read_overview()` returns `ExperimentOverview | None`.
  - Preserve no-auto-restore behavior for deleted experiments.

- [x] Update `automl/mlflow/project/overview.py::list_experiments()`.

  Required behavior:
  - Search ACTIVE_ONLY experiments through PyPI MLflow inside the seam.
  - Return logical experiment ids under the current route root, excluding `overview` and nested
    paths.
  - Respect current `namespace` and `dry_run` through `_routing.experiment_route(...)`.

- [x] Export experiment types/functions from `automl/experiment/__init__.py` and `automl/__init__.py`.

- [x] Run targeted tests.

  Run:
  ```bash
  uv run pytest tests/unit/experiment/test_overview_lifecycle.py tests/unit/mlflow/test_experiment_read_queries.py -v
  ```
  Expected: PASS.

**Acceptance:** Experiment state and listing primitives are in place for views and cleanup planning.

---

## P4.4 - Experiment Views: Leaderboard, Compare, Summary, Queries

**Files:**
- Create: `automl/experiment/views/types.py`
- Create: `automl/experiment/views/leaderboard.py`
- Create: `automl/experiment/views/compare.py`
- Create: `automl/experiment/views/queries.py`
- Create: `automl/experiment/views/summary.py`
- Modify: `automl/experiment/views/__init__.py`
- Modify: `automl/experiment/__init__.py`
- Test: `tests/unit/experiment/test_view_types.py`
- Test: `tests/unit/experiment/test_views.py`

**Specs:** `spec/09-experiment.md` Q3, Q6-Q7, Q10-Q13; `spec/10-trial.md` Q1-Q4.

**Legacy source:**
- `automl_legacy/inspect/views.py::leaderboard`
- `automl_legacy/inspect/views.py::compare`
- `automl_legacy/loop_context/queries.py::{recent_failures,strategies_attempted}`
- `automl_legacy/loop_context/summary.py`

**Migration rows:** `inspect/views.py::{LeaderboardRow,leaderboard,compare,experiments}`;
`loop_context/queries.py::{recent_failures,strategies_attempted}`; `loop_context/summary.py`.

**Steps:**
- [x] Write failing typed-view tests.

  Add `tests/unit/experiment/test_view_types.py` with tests equivalent to:
  ```python
  from automl.experiment.views.types import ComparisonResult, LeaderboardData, MetricDelta
  from automl.trial.types import TrialDetails, TrialSummary


  def test_leaderboard_data_from_dict_loads_trial_summaries():
      data = LeaderboardData.from_dict(
          {
              "metric": "test.auc",
              "experiment_id": "baseline",
              "rows": [{"run_id": "run-1", "status": "FINISHED"}],
              "n_unscored": 2,
          }
      )

      assert data.metric == "test.auc"
      assert data.n_unscored == 2
      assert isinstance(data.rows[0], TrialSummary)


  def test_comparison_result_from_dict_loads_details_and_metric_deltas():
      result = ComparisonResult.from_dict(
          {
              "run_ids": ["run-a", "run-b"],
              "runs": [{"run_id": "run-a"}, {"run_id": "run-b"}],
              "metric_deltas": [
                  {"metric": "test.auc", "value_a": 0.7, "value_b": 0.8, "delta": 0.1}
              ],
          }
      )

      assert isinstance(result.runs[0], TrialDetails)
      assert result.metric_deltas == (
          MetricDelta(metric="test.auc", value_a=0.7, value_b=0.8, delta=0.1),
      )
  ```

  Run:
  ```bash
  uv run pytest tests/unit/experiment/test_view_types.py -v
  ```
  Expected: FAIL because experiment view types do not exist.

- [x] Implement `automl/experiment/views/types.py`.

  Required behavior:
  - `LeaderboardData(schema_version=1, metric="", experiment_id="", rows=(), n_unscored=0)`.
  - `MetricDelta(metric="", value_a=None, value_b=None, delta=None)`.
  - `ComparisonResult(schema_version=1, run_ids=(), runs=(), metric_deltas=())`.
  - `from_dict` strips unknown keys and deserializes nested trial types.

- [x] Write failing view behavior tests.

  Add `tests/unit/experiment/test_views.py` with tests equivalent to:
  ```python
  import pytest

  from automl.eval import Auc, EvalSpec
  from automl.experiment import compare, leaderboard
  from automl.experiment.views.queries import recent_failures, strategies_attempted
  from automl.experiment.views.summary import build_summary, experiments
  from automl.mlflow import client, experiment, tags, trial
  from automl.project import ProjectConfig, Session


  @pytest.fixture
  def active(tmp_path):
      client.clear()
      config = ProjectConfig(
          project_name="home_credit",
          repo_root=tmp_path,
          project_dir=tmp_path / "projects" / "home_credit",
          eval_spec=EvalSpec(primary=Auc()),
          gcs_prefix="automl-root",
          mlflow_tracking_uri=(tmp_path / "mlruns").as_uri(),
      )
      session = Session(config=config, experiment_id="baseline")
      client.bind(
          tracking_uri=config.mlflow_tracking_uri,
          bucket="",
          gcs_prefix=config.gcs_prefix,
          project_name=config.project_name,
          experiment_id="baseline",
      )
      yield session
      client.clear()


  def _run(slug, metric=None, status="FINISHED", strategy="baseline"):
      experiment.ensure()
      if status == "FAILED":
          with pytest.raises(RuntimeError):
              with trial.active(slug=slug, strategy=strategy) as run_id:
                  raise RuntimeError("boom")
          return run_id
      with trial.active(slug=slug, strategy=strategy) as run_id:
          if metric is not None:
              trial.log_metric(run_id, "test.auc", metric)
              trial.log_metric(run_id, "auc", metric)
              trial.set_tag(run_id, tags.EVAL_PRIMARY_METRIC, "auc")
          return run_id


  def test_leaderboard_ranks_scored_trials_and_counts_unscored(active):
      low = _run("low", 0.61, strategy="linear")
      high = _run("high", 0.91, strategy="tree")
      _run("unscored", None, strategy="manual")

      data = leaderboard(n=5, session=active)

      assert [row.run_id for row in data.rows] == [high, low]
      assert data.n_unscored == 1
      assert data.metric == "auc"

      explicit = leaderboard(metric="test.auc", n=5, session=active)
      assert [row.run_id for row in explicit.rows] == [high, low]
      assert explicit.metric == "test.auc"


  def test_compare_returns_metric_deltas(active):
      left = _run("left", 0.7)
      right = _run("right", 0.85)

      result = compare([left, right], session=active)

      deltas = {item.metric: item for item in result.metric_deltas}
      assert deltas["test.auc"].delta == pytest.approx(0.15)
      assert [run.run_id for run in result.runs] == [left, right]


  def test_queries_and_summary_compose_over_seam(active):
      _run("failed", None, status="FAILED", strategy="tree")
      _run("scored", 0.8, strategy="linear")

      assert [row.strategy for row in recent_failures(session=active)] == ["tree"]
      assert strategies_attempted(session=active) == {"tree": 1, "linear": 1}

      summary = build_summary(session=active)
      assert summary["summary_kind"] == "experiment_summary"
      assert summary["trial_count"] == 2
      assert "learning_counts" not in summary
      assert experiments(session=active)[0]["experiment_id"] == "baseline"
  ```

  Run:
  ```bash
  uv run pytest tests/unit/experiment/test_views.py -v
  ```
  Expected: FAIL because view modules do not exist.

- [x] Implement `leaderboard()`.

  Required behavior:
  - Resolve session.
  - If `metric is None`, use the reviewed default `session.config.primary_metric` exactly.
  - Do not auto-discover or infer the default ranking metric from MLflow run contents.
  - Call `mlflow.experiment.top_n_by_metric(metric, n=n, training_origin=...)`.
  - Compute `n_unscored` from all filtered trial summaries minus the scored rows.
  - Return `LeaderboardData`.

- [x] Implement `compare()`.

  Required behavior:
  - Require at least one run id; pairwise deltas are computed over the first two run ids only.
  - Load each run through `trial.show_trial`.
  - Deltas cover the union of numeric `TrialDetails.metrics` keys; non-numeric/missing values
    produce `delta=None`.

- [x] Implement `recent_failures()` and `strategies_attempted()`.

  Required behavior:
  - `recent_failures(n=3, training_origin=None, session=None)` calls
    `mlflow.experiment.list_trials(status=TrialStatus.FAILED, limit=n, training_origin=...)`.
  - `strategies_attempted(session=None)` aggregates all trial summaries by `strategy`, including
    failed and unscored trials.

- [x] Implement `summary.py`.

  Required behavior:
  - `load_mlflow_context(session=None)` composes seam reads directly; it must not import `agent`.
  - `build_summary_from_context(context)` preserves the legacy keys except `learning_counts`.
  - `experiments(session=None)` enriches `mlflow.project.list_experiments()` with trial count and
    top metric data.

- [x] Export views from `automl/experiment/views/__init__.py` and `automl/experiment/__init__.py`.

- [x] Run targeted view tests.

  Run:
  ```bash
  uv run pytest tests/unit/experiment/test_view_types.py tests/unit/experiment/test_views.py -v
  ```
  Expected: PASS.

**Acceptance:** A4.1 and A4.3 are covered at unit level; A4.2 feeds compare through typed
`TrialDetails`.

---

## P4.5 - Cleanup Dataclasses And Dry-Run Plan Builder

**Files:**
- Create: `automl/project/cleanup.py`
- Test: `tests/unit/project/test_cleanup.py`

**Specs:** `spec/03-cleanup.md` sections 3-5, 7.8, 9-10, and 11.

**Legacy source:**
- `automl_legacy/cleanup.py::{CleanupPlan,build_cleanup_plan}`

**Migration rows:** `cleanup.py::{RouteCleanupTarget,MlflowDeleteResult,RunCleanupTarget,CleanupPlan,build_cleanup_plan}`.

**Steps:**
- [x] Write failing tests for cleanup report schemas and preview-only planning.

  Add `tests/unit/project/test_cleanup.py` with tests equivalent to:
  ```python
  from pathlib import Path

  from automl.project import ProjectConfig, Session
  from automl.project.cleanup import CleanupPlan, CleanupReport, delete


  def _session(tmp_path, *, dry_run=False, namespace="", experiment_id="baseline"):
      project_dir = tmp_path / "projects" / "home_credit"
      project_dir.mkdir(parents=True)
      return Session(
          config=ProjectConfig(
              project_name="home_credit",
              repo_root=tmp_path,
              project_dir=project_dir,
              gcs_bucket="automl-test-bucket",
              gcs_prefix="automl-root",
              mlflow_tracking_uri=(tmp_path / "mlruns").as_uri(),
          ),
          dry_run=dry_run,
          namespace=namespace,
          experiment_id=experiment_id,
      )


  def test_cleanup_report_from_dict_strips_unknown_fields():
      report = CleanupReport.from_dict(
          {
              "applied": False,
              "plan": {"scope": "experiment", "identifier": "baseline", "future": "ignored"},
              "future": "ignored",
          }
      )

      assert isinstance(report.plan, CleanupPlan)
      assert report.plan.scope == "experiment"
      assert report.result is None


  def test_experiment_delete_preview_builds_one_universe_plan_without_mutation(tmp_path):
      active = _session(tmp_path, dry_run=True, namespace="qa")
      local_root = (
          active.config.project_dir
          / "experiments"
          / "qa"
          / "dry_run"
          / "home_credit"
          / "baseline"
      )
      local_root.mkdir(parents=True)

      report = delete("baseline", scope="experiment", apply=False, session=active)

      assert report.applied is False
      assert report.result is None
      assert report.plan.dry_run is True
      assert report.plan.mlflow_experiment_targets == [
          ("qa/dry_run/home_credit/baseline", "")
      ]
      assert report.plan.gcs_prefix_patterns == [
          "gs://automl-test-bucket/automl-root/qa/dry_run/home_credit/baseline/"
      ]
      assert str(local_root) in report.plan.local_paths
      assert local_root.exists()


  def test_project_delete_plan_never_targets_user_authored_project_dir(tmp_path):
      active = _session(tmp_path)
      report = delete("home_credit", scope="project", apply=False, session=active)

      assert str(active.config.project_dir) not in report.plan.local_paths
      assert all("/experiments/" in path or "/.cache/" in path for path in report.plan.local_paths)
  ```

  Run:
  ```bash
  uv run pytest tests/unit/project/test_cleanup.py -v
  ```
  Expected: FAIL because `project.cleanup` does not exist.

- [x] Implement cleanup schemas in `automl/project/cleanup.py`.

  Required behavior:
  - Frozen `CleanupPlan`, `CleanupResult`, `CleanupReport` with fields from spec 03 section 10.
  - `from_dict` loaders strip unknown keys and load nested plan/result objects.
  - Keep `CleanupPlan.gcs_prefix_patterns` as prefix strings, not enumerated blobs.

- [x] Implement preview plan construction.

  Required behavior:
  - Public `delete(name, *, scope="project", apply=False, hard_delete=False, session=None)`
    is the project-domain engine entry point.
  - Private `_build_plan(scope, identifier, session)` builds one exact route using
    `session.namespace`, `session.dry_run`, `session.project_name`, and the scope identifier.
  - Experiment scope plan targets exactly one MLflow experiment route, one GCS prefix, and local
    proposal/timeline/session-lock/trial-root paths for that route.
  - Project scope plan targets the overview route and all logical child routes; if no MLflow state
    exists yet, it still includes the project route GCS/local prefixes so orphan blobs and local
    dirs are catchable.
  - Trial scope accepts only a run-id plus a resolved `ParentExperimentRef` supplied by the trial
    wrapper in P4.7; it never accepts a slug selector.
  - `apply=False` never calls MLflow delete, GCS delete, or `shutil.rmtree`.

- [x] Keep project cleanup exports scoped to `automl.project.cleanup`.

  Required behavior:
  - Do not modify `automl/project/__init__.py` in Phase 4.
  - Public user-facing cleanup for A4 is `automl.experiment.delete`; project cleanup remains the
    shared engine module until the Phase 6 CLI/catalog pass.

- [x] Run cleanup plan tests.

  Run:
  ```bash
  uv run pytest tests/unit/project/test_cleanup.py -v
  ```
  Expected: PASS for schema and preview-plan tests.

**Acceptance:** Cleanup has dry-run/idempotent planning before any destructive implementation.

---

## P4.6 - Cleanup Apply Engine, GCS Delete Helpers, And Hard Delete

**Files:**
- Modify: `automl/project/cleanup.py`
- Modify: `automl/utils/io/gcs.py`
- Test: `tests/unit/project/test_cleanup.py`
- Test: `tests/unit/utils/test_gcs.py`

**Specs:** `spec/03-cleanup.md` sections 6-7, 10-11.

**Legacy source:**
- `automl_legacy/cleanup.py::{apply_cleanup_plan,_delete_gcs_prefix,_delete_mlflow_experiment,_delete_mlflow_run,_run_mlflow_gc}`

**Migration rows:** `cleanup.py::apply_cleanup_plan`.

**Steps:**
- [x] Extend failing cleanup tests for apply, idempotency, per-blob GCS statuses, and hard-delete command.

  Add cases equivalent to:
  ```python
  import subprocess

  from automl.mlflow import client, experiment
  from automl.project.cleanup import CleanupResult, delete
  from automl.utils.io import gcs


  class DeleteBlob:
      def __init__(self, name, deleted):
          self.name = name
          self._deleted = deleted

      def delete(self):
          self._deleted.append(self.name)


  class DeleteClient:
      def __init__(self, names):
          self.deleted = []
          self._names = names

      def list_blobs(self, bucket, prefix):
          assert bucket == "automl-test-bucket"
          return [DeleteBlob(name, self.deleted) for name in self._names if name.startswith(prefix)]


  def test_gcs_delete_prefix_collects_deleted_count():
      fake = DeleteClient(["automl-root/home_credit/baseline/a.json", "other/x.json"])

      result = gcs.delete_prefix(
          "gs://automl-test-bucket/automl-root/home_credit/baseline/",
          client=fake,
      )

      assert result == 1
      assert fake.deleted == ["automl-root/home_credit/baseline/a.json"]


  def test_apply_soft_deletes_mlflow_then_gcs_then_local(tmp_path, monkeypatch):
      active = _session(tmp_path)
      client.bind(
          tracking_uri=active.config.mlflow_tracking_uri,
          bucket=active.config.gcs_bucket,
          gcs_prefix=active.config.gcs_prefix,
          project_name=active.project_name,
          experiment_id=active.active_experiment_id,
      )
      experiment.ensure()
      local_root = active.config.project_dir / "experiments" / "home_credit" / "baseline"
      local_root.mkdir(parents=True)
      fake = DeleteClient(["automl-root/home_credit/baseline/runs/run-1/model.pkl"])
      monkeypatch.setattr(gcs, "_gcs_client", lambda: fake)

      report = delete("baseline", scope="experiment", apply=True, session=active)

      assert report.applied is True
      assert isinstance(report.result, CleanupResult)
      assert report.result.mlflow_experiments["home_credit/baseline"] == "deleted"
      assert report.result.gcs[
          "gs://automl-test-bucket/automl-root/home_credit/baseline/"
      ] == 1
      assert report.result.local[str(local_root)] == "deleted"
      assert not local_root.exists()

      rerun = delete("baseline", scope="experiment", apply=True, session=active)
      assert rerun.applied is True
      assert rerun.result.gcs[
          "gs://automl-test-bucket/automl-root/home_credit/baseline/"
      ] == 0


  def test_hard_delete_runs_mlflow_gc_after_soft_delete(tmp_path, monkeypatch):
      active = _session(tmp_path)
      client.bind(
          tracking_uri=active.config.mlflow_tracking_uri,
          bucket=active.config.gcs_bucket,
          gcs_prefix=active.config.gcs_prefix,
          project_name=active.project_name,
          experiment_id=active.active_experiment_id,
      )
      experiment.ensure()
      monkeypatch.setenv("MLFLOW_BACKEND_STORE_URI", "sqlite:////tmp/mlflow.db")
      monkeypatch.setattr(gcs, "_gcs_client", lambda: DeleteClient([]))
      calls = []

      def fake_run(command, check, capture_output, text):
          calls.append(command)
          return subprocess.CompletedProcess(command, 0, stdout="gc complete", stderr="")

      monkeypatch.setattr(subprocess, "run", fake_run)

      report = delete("baseline", scope="experiment", apply=True, hard_delete=True, session=active)

      assert report.result.mlflow_hard_delete_status == "success"
      assert calls and calls[0][:4] == ["uv", "run", "mlflow", "gc"]
  ```

  Run:
  ```bash
  uv run pytest tests/unit/project/test_cleanup.py tests/unit/utils/test_gcs.py -v
  ```
  Expected: FAIL because apply/delete helpers are not implemented.

- [x] Add GCS delete helpers to `automl/utils/io/gcs.py`.

  Required behavior:
  - `delete_prefix(uri, *, client=None) -> int | str` lists blobs under the prefix and deletes
    each blob inside its own `try/except`.
  - If all deletes succeed, return deleted count.
  - If any blob fails, return a stable `"failed: ..."` string containing enough detail for the
    cleanup result; do not abort the whole cascade on one blob.
  - Keep existing read/write helper behavior unchanged.

- [x] Implement cleanup apply engine.

  Required behavior:
  - `_apply_plan(plan, *, hard_delete, session) -> CleanupResult`.
  - MLflow first: soft-delete active experiments/runs. Already-deleted/not-found records are
    non-errors.
  - GCS second: call `gcs.delete_prefix()` for each prefix pattern.
  - Local third: `shutil.rmtree` each local path, with missing paths recorded as
    `"skipped: not found"`.
  - Empty plans return a no-op result with empty dicts.
  - Per-target failures are collected; systemic list/connect failures raise `StorageError`.

- [x] Implement hard-delete command.

  Required behavior:
  - Resolve backend store URI from `MLFLOW_BACKEND_STORE_URI`; optionally auto-detect
    `mlflow_local/mlflow.db` under repo root or its parent.
  - Run `uv run mlflow gc --backend-store-uri <uri> --experiment-ids <comma ids>`.
  - Add `--artifacts-destination <session.config.mlflow_artifacts_destination>` when non-empty.
  - Record `mlflow_hard_delete_status` and raw output in `CleanupResult`.
  - If direct backend access is unavailable, raise `StorageError` with the spec's actionable
    message.

- [x] Run cleanup apply tests.

  Run:
  ```bash
  uv run pytest tests/unit/project/test_cleanup.py tests/unit/utils/test_gcs.py -v
  ```
  Expected: PASS.

**Acceptance:** Cleanup destructive behavior is covered by fake/local tests before the external
gate can delete anything.

---

## P4.7 - Experiment And Trial Cleanup Wrappers With Integration Safety Tests

**Files:**
- Create: `automl/experiment/cleanup.py`
- Create: `automl/trial/cleanup.py`
- Modify: `automl/experiment/__init__.py`
- Modify: `automl/trial/__init__.py`
- Test: `tests/integration/cleanup/test_experiment_delete.py`
- Test: `tests/unit/project/test_cleanup.py`

**Specs:** `spec/03-cleanup.md` sections 1, 3, 4.3, 9.1-9.2; `spec/09-experiment.md` Q10-Q11;
`spec/10-trial.md` Q8.

**Legacy source:**
- `automl_legacy/trial/cleanup.py`
- `automl_legacy/cleanup.py`

**Migration rows:** `trial/cleanup.py::cleanup`; `cleanup.py::{run,_build_parser}` remains CLI-deferred.

**Steps:**
- [x] Write failing wrapper and isolation integration tests.

  Add `tests/integration/cleanup/test_experiment_delete.py` with tests equivalent to:
  ```python
  from pathlib import Path

  import pytest

  from automl.experiment import delete as delete_experiment
  from automl.mlflow import client, experiment, trial
  from automl.project import ProjectConfig, Session
  from automl.trial.cleanup import delete as delete_trial
  from automl.utils.io import gcs


  class FakeBlob:
      def __init__(self, store, bucket, name):
          self._store = store
          self._bucket = bucket
          self.name = name

      def delete(self):
          self._store.pop((self._bucket, self.name), None)

      def upload_from_string(self, data, **kwargs):
          self._store[(self._bucket, self.name)] = data if isinstance(data, bytes) else data.encode()


  class FakeBucket:
      def __init__(self, store, name):
          self._store = store
          self.name = name

      def blob(self, name):
          return FakeBlob(self._store, self.name, name)

      def list_blobs(self, prefix):
          return [
              FakeBlob(self._store, bucket, name)
              for (bucket, name) in sorted(self._store)
              if bucket == self.name and name.startswith(prefix)
          ]


  class FakeGCSClient:
      def __init__(self):
          self.store = {}

      def bucket(self, name):
          return FakeBucket(self.store, name)

      def list_blobs(self, bucket, prefix):
          return self.bucket(bucket).list_blobs(prefix)


  def _active(tmp_path, *, dry_run=False, namespace="", experiment_id="baseline"):
      project_dir = tmp_path / "projects" / "home_credit"
      project_dir.mkdir(parents=True)
      active = Session(
          config=ProjectConfig(
              project_name="home_credit",
              repo_root=tmp_path,
              project_dir=project_dir,
              gcs_bucket="automl-test-bucket",
              gcs_prefix="automl-root",
              mlflow_tracking_uri=(tmp_path / "mlruns").as_uri(),
          ),
          dry_run=dry_run,
          namespace=namespace,
          experiment_id=experiment_id,
      )
      client.bind(
          tracking_uri=active.config.mlflow_tracking_uri,
          bucket=active.config.gcs_bucket,
          gcs_prefix=active.config.gcs_prefix,
          project_name=active.project_name,
          experiment_id=experiment_id,
          dry_run=dry_run,
          namespace=namespace,
      )
      return active


  def test_experiment_delete_apply_removes_only_current_universe(tmp_path, monkeypatch):
      fake = FakeGCSClient()
      monkeypatch.setattr(gcs, "_gcs_client", lambda: fake)
      active = _active(tmp_path, namespace="qa", experiment_id="phase4")
      sibling = _active(tmp_path, namespace="prod", experiment_id="phase4")
      active = _active(tmp_path, namespace="qa", experiment_id="phase4")

      experiment.ensure()
      with trial.active(slug="scored", strategy="baseline") as run_id:
          trial.log_metric(run_id, "test.auc", 0.8)
      fake.store[("automl-test-bucket", "automl-root/qa/home_credit/phase4/runs/blob.json")] = b"{}"
      fake.store[("automl-test-bucket", "automl-root/prod/home_credit/phase4/runs/blob.json")] = b"{}"
      local_root = active.config.project_dir / "experiments" / "qa" / "home_credit" / "phase4"
      sibling_root = sibling.config.project_dir / "experiments" / "prod" / "home_credit" / "phase4"
      local_root.mkdir(parents=True)
      sibling_root.mkdir(parents=True)

      report = delete_experiment("phase4", apply=True, session=active)

      assert report.applied is True
      assert ("automl-test-bucket", "automl-root/qa/home_credit/phase4/runs/blob.json") not in fake.store
      assert ("automl-test-bucket", "automl-root/prod/home_credit/phase4/runs/blob.json") in fake.store
      assert not local_root.exists()
      assert sibling_root.exists()


  def test_trial_delete_rejects_run_from_other_namespace(tmp_path):
      qa = _active(tmp_path, namespace="qa", experiment_id="phase4")
      experiment.ensure()
      with trial.active(slug="qa-run", strategy="baseline") as run_id:
          pass
      prod = _active(tmp_path, namespace="prod", experiment_id="phase4")

      with pytest.raises(Exception, match="current session"):
          delete_trial(run_id, apply=False, session=prod)
  ```

  Run:
  ```bash
  uv run pytest tests/integration/cleanup/test_experiment_delete.py -v
  ```
  Expected: FAIL because cleanup wrappers do not exist.

- [x] Implement `automl/experiment/cleanup.py`.

  Required behavior:
  - `delete(experiment_id, *, apply=False, hard_delete=False, session=None)` resolves session and
    delegates to `project.cleanup.delete(..., scope="experiment")`.

- [x] Implement `automl/trial/cleanup.py`.

  Required behavior:
  - Resolve session.
  - Call `mlflow.trial.get_parent_experiment(run_id)`.
  - Verify parent project equals session project.
  - Verify parent route exactly matches the route for this session's namespace/dry_run/project and
    parent experiment id.
  - Delegate to the project cleanup engine with scope `"trial"` and run id target.

- [x] Extend the project cleanup engine for trial scope.

  Required behavior:
  - MLflow target is the one run id.
  - GCS prefix is that run's canonical `run_bulk` prefix. Use run start time when available so
    Phase 3's month-partitioned GCS layout is targeted accurately; if unavailable, fall back to
    the current partition and rely on experiment-scope cleanup for broader prefix deletion.
  - Local path is the matching trial sandbox under `runner.paths.route_root(session)`.
  - Trial delete never mutates experiment overview tags.

- [x] Export wrapper functions.

- [x] Run integration cleanup tests.

  Run:
  ```bash
  uv run pytest tests/integration/cleanup/test_experiment_delete.py tests/unit/project/test_cleanup.py -v
  ```
  Expected: PASS.

**Acceptance:** A4.4 cleanup safety is covered locally, including namespace isolation and rerun behavior.

---

## P4.8 - Phase 4 E2E Gate And Phase 3 Regression

**Files:**
- Create: `tests/e2e/test_phase4_experiment_trial_cleanup.py`
- Modify: no production files unless gate failures expose a real implementation gap.

**Specs:** `plan/acceptance-checklist.md` A4.1-A4.4; `README.md` harness instructions.

**Steps:**
- [x] Write the opt-in Phase 4 external gate.

  Add `tests/e2e/test_phase4_experiment_trial_cleanup.py` with the scenario:
  ```python
  import os
  from datetime import UTC, datetime
  from pathlib import Path

  import pytest

  from automl.data import materialize
  from automl.experiment import compare, delete as delete_experiment, leaderboard
  from automl.experiment.views.queries import recent_failures, strategies_attempted
  from automl.experiment.views.summary import build_summary
  from automl.mlflow import client as mlflow_client
  from automl.mlflow import experiment as mlflow_experiment
  from automl.mlflow import trial as mlflow_trial
  from automl.project import clear_session, use_project
  from automl.runner import TrialStatus, run_trial
  from automl.trial import load_model, show_trial
  from automl.utils.io import gcs


  @pytest.mark.skipif(
      not (
          os.environ.get("AUTOML_PHASE4_E2E")
          and os.environ.get("GCS_BUCKET")
          and os.environ.get("MLFLOW_TRACKING_URI")
          and os.environ.get("GCP_PROJECT")
      ),
      reason="Phase 4 e2e requires AUTOML_PHASE4_E2E, GCS_BUCKET, GCP_PROJECT, and MLFLOW_TRACKING_URI",
  )
  def test_phase4_experiment_trial_cleanup_gate():
      repo_root = Path(__file__).resolve().parents[2]
      stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
      namespace = f"qa-phase4-{stamp}"
      experiment_id = f"phase4-{stamp}"
      active = use_project(
          "example_homecredit",
          repo_root=repo_root,
          namespace=namespace,
          experiment_id=experiment_id,
      )
      try:
          materialize(session=active)
          first = run_trial("example_homecredit", session=active)
          second = run_trial("example_homecredit", session=active)
          assert first.status is TrialStatus.FINISHED
          assert second.status is TrialStatus.FINISHED

          with pytest.raises(RuntimeError):
              with mlflow_trial.active(slug="phase4_failed", strategy="forced_failure") as failed_run_id:
                  raise RuntimeError("phase4 forced failure")
          with mlflow_trial.active(slug="phase4_unscored", strategy="unscored") as unscored_run_id:
              pass

          board = leaderboard(session=active, n=10)
          assert board.rows
          assert board.metric == active.config.primary_metric
          assert first.run_id in {row.run_id for row in board.rows}
          assert second.run_id in {row.run_id for row in board.rows}
          assert board.n_unscored >= 2

          comparison = compare([first.run_id, second.run_id], session=active)
          assert comparison.metric_deltas
          assert [run.run_id for run in comparison.runs] == [first.run_id, second.run_id]

          details = show_trial(first.run_id, session=active)
          assert details.evaluations
          assert load_model(first.run_id) is not None

          summary = build_summary(session=active)
          assert summary["trial_count"] >= 4
          assert recent_failures(session=active)
          assert strategies_attempted(session=active)

          route_prefix = (
              f"gs://{active.config.gcs_bucket}/{active.config.gcs_prefix}/"
              f"{namespace}/{active.project_name}/{experiment_id}/"
          )
          assert gcs.list_blob_names(route_prefix)
          report = delete_experiment(experiment_id, apply=True, session=active)
          assert report.applied is True
          assert gcs.list_blob_names(route_prefix) == []
          assert mlflow_client.raw().get_experiment_by_name(
              f"{namespace}/{active.project_name}/{experiment_id}"
          ).lifecycle_stage == "deleted"
      finally:
          clear_session()
  ```

  Before implementing, review whether this should use direct library calls only or literal CLI
  wrappers. If literal CLI is required, amend this plan before adding the CLI.

- [x] Run the Phase 4 e2e gate against the original MLflow server.

  Run after loading the worktree `.env`:
  ```bash
  AUTOML_PHASE4_E2E=1 MLFLOW_TRACKING_URI=http://127.0.0.1:54321 uv run pytest tests/e2e/test_phase4_experiment_trial_cleanup.py -v
  ```
  Expected: PASS. It must use a unique `qa-phase4-*` namespace plus unique `experiment_id`, then
  soft-delete only that experiment.

- [x] Re-run the Phase 3 external-eval gate.

  Run after loading the worktree `.env`:
  ```bash
  AUTOML_PHASE3_E2E=1 MLFLOW_TRACKING_URI=http://127.0.0.1:54321 uv run pytest tests/e2e/test_phase3_eval_breadth.py -v
  ```
  Expected: PASS.

- [x] Run the full non-external suite and ratchets.

  Run:
  ```bash
  uv run pytest tests/unit tests/contracts tests/integration -v
  uv run pytest tests/contracts -v
  rg 'automl_legacy' automl projects tests
  rg '(^|\s)(import|from) mlflow' automl projects tests
  ```
  Expected: all tests pass; import ratchets remain clean.

**Acceptance:** A4.1-A4.4 are proven externally and Phase 3 remains green.

---

## P4.9 - Docs Closeout And Commit

**Files:**
- Modify: `docs/superpowers/automl-refactor/README.md`
- Modify: `docs/superpowers/automl-refactor/plan/README.md`
- Modify: `docs/superpowers/automl-refactor/plan/implementation-strategy.md` if boundaries changed
- Modify: `docs/superpowers/automl-refactor/plan/acceptance-checklist.md`
- Modify: `docs/superpowers/automl-refactor/plan/migration-checklist.md`
- Modify: `docs/superpowers/automl-refactor/plan/phases/phase-4-experiment-trial-cleanup.md`

**Specs:** plan folder working norms; implementation strategy section 5.

**Steps:**
- [x] Update the Phase 4 plan with actual completion evidence.

  Required evidence lines:
  ```text
  uv run pytest tests/unit tests/contracts tests/integration -v -> <actual result>
  AUTOML_PHASE4_E2E=1 MLFLOW_TRACKING_URI=http://127.0.0.1:54321 uv run pytest tests/e2e/test_phase4_experiment_trial_cleanup.py -v -> <actual result>
  AUTOML_PHASE3_E2E=1 MLFLOW_TRACKING_URI=http://127.0.0.1:54321 uv run pytest tests/e2e/test_phase3_eval_breadth.py -v -> <actual result>
  uv run pytest tests/contracts -v -> <actual result>
  ```

- [x] Flip acceptance checklist A4 rows only after the gates pass.

  Required changes:
  - A4.1 -> `[x]`
  - A4.2 -> `[x]`
  - A4.3 -> `[x]`
  - A4.4 -> `[x]`

- [x] Flip migration checklist rows for code that actually landed.

  Required Phase 4 rows expected to flip:
  - `cleanup.py` dataclasses and private plan/apply functions.
  - `inspect/views.py::leaderboard`, `show_trial`, `compare`, `experiments`, `load_model`.
  - `loop_context/queries.py::top_n_by_metric`, `recent_failures`, `strategies_attempted`,
    and `show_trial` disposition.
  - `loop_context/summary.py` functions.
  - `mlflow/store.py::ensure_experiment_overview`, `get_trial_summaries`, and read URL helpers
    if touched.
  - `trial/cleanup.py::cleanup`.

  Required rows to leave open unless implemented by evidence:
  - `trial/create.py`, `trial/fork.py`, `trial/promote.py`, `trial/packaging.py`.
  - Full CLI rows.
  - Agent/proposer-context rows.
  - Notebook/example cleanup rows.

- [x] Update README and plan README current status.

  Required status:
  - Phase 4 complete with commit id after commit.
  - Next action becomes writing/reviewing Phase 5 detailed plan.

- [x] Run stale-status scans.

  Run:
  ```bash
  rg "Phase 4|A4|NEXT ACTION|not started|in progress|phase-4" docs/superpowers/automl-refactor
  rg "eval_snapshot" projects docs/superpowers/automl-refactor
  ```
  Expected: no stale Phase 4 "next action" language after closeout; remaining notebook
  `eval_snapshot` hits are documented as deferred and not part of Phase 4.

- [x] Commit Phase 4 after docs and verification.

  Run:
  ```bash
  git status --short
  git add automl tests docs/superpowers/automl-refactor
  git commit -m "Complete Phase 4 experiment trial cleanup"
  ```
  Expected: commit succeeds on branch `refactor/four-layer`.

**Acceptance:** Phase 4 is closed with tests, docs, checklist evidence, and a commit.

---

## Carryover Notes For Later Phases

- Phase 6 CLI should wrap the Phase 4 domain/library functions without changing their semantics:
  `experiment leaderboard` defaults to the config-defined primary metric, exposes a `--metric`
  override, and reports the resolved key.
- Phase 6 namespace/dry-run breadth should reuse the Phase 4 cleanup route-validation rules and
  the QA namespace testing pattern. Do not weaken the exact-route check when adding broader
  command coverage.
- Phase 6 CLI names should match the spec command catalog; Phase 4 intentionally keeps only the
  library destinations that those commands will call.
- Phase 7 cutover should verify that no later CLI/plugin wrapper reintroduces metric
  auto-discovery from historical MLflow runs.

---

## Deferred Out Of Phase 4

- Full CLI dispatcher and all noun commands: Phase 6; Phase 4 only builds their domain/library
  destinations.
- Agent proposer context, launcher, timeline, hooks, and proposal validation: Phase 5.
- Trial create/fork/promote/package_model: keep open unless a Phase 4 gate failure proves one is
  required.
- Broad namespace/dry-run surface outside cleanup safety: Phase 6.
- Eval checks migration and eval CLI surfaces: later phase unless a Phase 4 read gate requires them.
- Old example notebooks with `eval_snapshot` references: deferred unless notebook/user surface is
  touched directly.
- Cross-experiment orphan scanning and admin `mlflow gc` verb: follow-demand additions from spec 03.
