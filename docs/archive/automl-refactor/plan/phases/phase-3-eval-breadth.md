# Phase 3 Eval Breadth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development`
> (recommended) or `superpowers:executing-plans` to implement this plan task-by-task.
> Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** thicken the proven Phase 2 path enough to pass A3.1-A3.2: full eval metric
breadth, durable split-view and external EvalDataset handling, Augmentation support,
Predictions and EvalIndex persistence, and multi-label eval round-trips.

**Architecture:** keep the runner as the straight-line fit owner while moving eval persistence
back into the stateful `evaluate()` verb, as specified by 07/08. Eval owns eval-data loading and
eval artifacts; data remains the public slice loader for split views; MLflow remains the only
package that imports PyPI `mlflow`. The Phase 3 model-load addition is seam-local
(`mlflow.trial.artifacts.load_model`) so post-hoc external eval can use the trial's persisted
model without adding Phase 4 `trial.show` / `trial.load_model` read models.

**Tech Stack:** Python 3.11 via `uv`; pytest; pandas/numpy/pyarrow; scikit-learn metrics;
cloudpickle; original local MLflow at `http://127.0.0.1:54321` with this worktree's `.env`
loaded; GCS bucket `gs://automl-homecredit-kaggle-wliu`.

**Acceptance:** `plan/acceptance-checklist.md` rows **A3.1-A3.2**.

---

## Review Checkpoint

This plan was reviewed after Phase 2 commit `bc3a2ef` and implemented in Phase 3. The task order
below is evidence-cut from spec 07, specs 00/01/02/05/06/08/10, the Phase 2 code, and the legacy
eval flow.

**Closeout (2026-05-28):** Phase 3 landed and A3.1-A3.2 passed. Evidence:
`uv run pytest tests/unit tests/contracts tests/integration -v` -> `191 passed`;
`AUTOML_PHASE3_E2E=1 MLFLOW_TRACKING_URI=http://127.0.0.1:54321 uv run pytest tests/e2e/test_phase3_eval_breadth.py -v` -> `1 passed`;
`uv run pytest tests/contracts -v` -> `9 passed`. Deferred consciously: broad
`eval/checks.py` migration, CLI eval surfaces, experiment/trial read models, leaderboard/compare,
cleanup cascade, public trial-domain `load_model`, and old example notebooks that still reference
the pre-refactor `eval_snapshot` API. `utils.io.gcs.delete()` was not needed for Phase 3 because
partial-write rollback did not require a delete helper.

## Evidence Read

- `spec/07-eval.md`: eval owns `Metric`, `EvalSpec`, `EvalDataset`, `Augmentation`,
  `evaluate`, `EvalResult`, `EvalIndex`, and `Predictions`. `split_view` is recipe-only and
  delegates realization to `data.load_dataset_by_id(dataset_id, split_range=buckets)`. `external`
  owns parquet bytes. `evaluate()` persists predictions, eval results, EvalIndex, and scalar
  MLflow metrics.
- `spec/02-mlflow-seam.md`: eval and predictions are multi-instance trial artifacts keyed by
  `label`; `list_eval(run_id)` is tag-backed and returns `(label, eval_dataset_id)` pairs.
- `spec/08-runner.md`: runner loads only the fit slice; eval owns eval-data loading. Runner may
  call `prepare_eval_dataset(session=active, dataset_id=loaded_fit.id, split=eval_split)` and
  `evaluate(session=active, model_run_id=run_id, eval_dataset_id=eval_dataset.id)`, passing `_model` and
  `_model_feature_registry` while the fitted model is in memory.
- Current Phase 2 code: `automl/eval` has only `Auc`, process-local split-view recipes,
  no durable eval manifests, no `LogLoss`/`ThresholdSweep`, no `Predictions`/`EvalIndex`, and
  the runner currently logs eval artifacts and metrics after `evaluate()`.
- Legacy source: `automl_legacy/eval/base.py` contains alias/sign/augmentation/scalar helpers;
  `automl_legacy/eval/evaluate.py` contains metric caching, predictions writing, EvalIndex-like
  TOC, scalar metric logging, and augmentation loading; `automl_legacy/eval/publish.py` contains
  external eval and augmentation validation/publish behavior.

## Planning Interpretations

- `ThresholdSweep` returns a non-scalar value. Persist it in `EvalResult.metrics`; do not log it
  as an MLflow scalar. This follows legacy `scalar_metric_records` and MLflow's scalar metric
  constraint. A3.1's namespaced MLflow logging is verified for finite scalar metrics in the
  locked set (`auc`, `negative_log_loss`, and scalar custom augmentation metrics).
- Persist `EvalIndex` as `eval/manifest.json` for rich round-trip and future reads, while
  keeping `mlflow.trial.artifacts.list_eval(run_id)` tag-backed per spec 02.
- Cache state is runtime metadata, not durable identity and not part of serialized contracts.
  Prepare helpers may return `(object, cached)` for local runtime behavior, and `EvalResult.cached`
  may describe whether `evaluate()` reused existing artifacts, but no manifest/result payload
  stores `cached`.
- Cache reuse is opportunistic: when complete matching artifacts already exist and
  `overwrite=False`, load and return them; when artifacts are absent, partial, invalid, or
  `overwrite=True`, recompute and rewrite.
- Full `eval/checks.py` migration is outside the A3 gate unless implementation evidence requires
  it. Implement the compatibility/column checks needed by `prepare_*` and `evaluate()` locally in
  Phase 3, and leave any broader validation/checks migration row open for a later phase.
- Do not add Phase 4 surfaces: no leaderboard/compare, no `trial.show_trial`, no trial-domain
  `load_model`, no cleanup cascade, no CLI breadth.

## Non-Negotiable Decisions

- New code must never import `automl_legacy`.
- Only `automl/mlflow/**` may import the PyPI `mlflow` package.
- Use `uv` for all commands.
- Write failing tests before implementation in every task.
- Keep the Phase 2 WOE-gated Home Credit path green after each task.
- Use the original MLflow server for external gates:
  `MLFLOW_TRACKING_URI=http://127.0.0.1:54321` after loading this worktree's `.env`.
- Do not broaden into experiment/trial views, cleanup, agent loop, CLI catalog, namespace/dry-run
  breadth, or cutover.

## Migration Rows Covered

- `eval/base.py`: `is_scalar_value`, `Metric`, `EvalSpec`, `scalar_metric_records`.
- `eval/metrics.py`: `Auc`, `LogLoss`, `ThresholdSweep`.
- `eval/results.py`: `EvalResult`, `EvalIndex`, `Predictions`.
- `eval/_load.py` and `eval/eval_dataset.py`: `LoadedEvalDataset`, `load_eval_dataset`,
  `EvalDataset`, `Augmentation`, identity helpers, validators.
- `eval/prepare.py`: `prepare_eval_dataset`, `prepare_eval_augmentation`,
  split-view prepare path.
- `mlflow/artifacts/eval.py`: `validate_eval_label`, `write_evaluation_results` ->
  `mlflow/trial/artifacts/eval.py::{write_eval,load_eval,list_eval,write_eval_index,load_eval_index}`.
- `mlflow/artifacts/predictions.py`: `PredictionsArtifact`, `write_predictions_gcs` ->
  `mlflow/trial/artifacts/predictions.py::{write_predictions,load_predictions,list_predictions}`.
- `runner/_execute.py`: eval result consumption and `_model` / `_model_feature_registry`
  injection remain in `runner/trial.py`.

## Task DAG

```
P3.0 baseline guard
  -> P3.1 metric and EvalSpec breadth
  -> P3.2 EvalResult, EvalIndex, Predictions schemas
  -> P3.3 MLflow eval and predictions artifacts
  -> P3.4 durable EvalDataset load/prepare for split_view and external
  -> P3.5 Augmentation publish/load and metric integration
  -> P3.6 evaluate() owns persistence, caching-light scalar logging, and post-hoc model load
  -> P3.7 runner integration and Phase 2 regression guard
  -> P3.8 Phase 3 e2e gate
  -> P3.9 docs closeout
```

Each task is a review boundary. If specs contradict running evidence, stop and flag the
contradiction before changing code.

---

## P3.0 - Baseline Guard And Scope Ratchet

**Files:**
- Modify: no production files.
- Test/read: existing Phase 2 tests.

**Specs:** `plan/implementation-strategy.md` section 5-6; `plan/acceptance-checklist.md`
A2/A3.

**Steps:**
- [x] Run the Phase 2 non-external suite before changing code.

  Run:
  ```bash
  uv run pytest tests/unit tests/contracts tests/integration -v
  ```
  Expected: PASS. If this fails, fix the regression before starting Phase 3.

- [x] Run the architecture ratchets.

  Run:
  ```bash
  uv run pytest tests/contracts -v
  rg 'automl_legacy' automl projects tests
  rg '(^|\s)(import|from) mlflow' automl projects tests
  ```
  Expected: contract tests pass; `automl_legacy` appears only in ratchet/doc text; PyPI
  `mlflow` imports appear only under `automl/mlflow/` or tests intentionally verifying the
  seam.

- [x] Leave A3 rows unchecked. This task establishes the starting line only.

**Acceptance:** Phase 2 remains green before Phase 3 changes begin.

---

## P3.1 - Metric And EvalSpec Breadth

**Files:**
- Modify: `automl/eval/base.py`
- Modify: `automl/eval/metrics.py`
- Modify: `automl/eval/__init__.py`
- Test: `tests/unit/eval/test_metrics_breadth.py`
- Test: `tests/unit/eval/test_eval_thin_path.py`

**Specs:** `spec/07-eval.md` "Remaining surface"; `spec/02-mlflow-seam.md` section 6.2.2
and section 10.

**Legacy source:**
- `automl_legacy/eval/base.py`
- `automl_legacy/eval/metrics.py`
- `automl_legacy/eval/runner.py`

**Migration rows:** `eval/base.py::{is_scalar_value,Metric,EvalSpec,scalar_metric_records}`;
`eval/metrics.py::{Auc,LogLoss,ThresholdSweep}`.

**Steps:**
- [x] Write failing tests for metric names, aliases, signing, required augmentations, and scalar
  extraction.

  Add `tests/unit/eval/test_metrics_breadth.py` with tests equivalent to:
  ```python
  import pandas as pd
  import pytest

  from automl.eval import Auc, EvalSpec, LogLoss, ThresholdSweep
  from automl.eval.base import Metric, is_scalar_value, scalar_metric_records


  class RevenueLift(Metric):
      name = "revenue_lift"
      required_columns = ("amount",)
      required_augmentations = ("risk_weight",)

      def compute(self, df, y_pred, target_col):
          return float((df["amount"] * df["risk_weight"] * y_pred).mean())


  def _frame():
      return pd.DataFrame(
          {
              "target": [0, 0, 1, 1],
              "amount": [10.0, 20.0, 30.0, 40.0],
              "risk_weight": [1.0, 1.1, 1.2, 1.3],
          }
      )


  def test_builtin_metrics_and_threshold_sweep_values():
      df = _frame()
      y_pred = pd.Series([0.05, 0.25, 0.75, 0.95])
      result = EvalSpec(
          primary=Auc(),
          metrics=[-LogLoss(), ThresholdSweep(thresholds=[0.25, 0.75])],
      ).evaluate(df, y_pred, "target")

      by_name = {record["name"]: record["value"] for record in result["metrics"]}
      assert result["primary"] == "auc"
      assert by_name["auc"] == pytest.approx(1.0)
      assert by_name["negative_log_loss"] < 0.0
      assert by_name["threshold_sweep"] == [
          {"threshold": 0.25, "precision": pytest.approx(2 / 3), "recall": pytest.approx(1.0)},
          {"threshold": 0.75, "precision": pytest.approx(1.0), "recall": pytest.approx(1.0)},
      ]


  def test_alias_sign_and_required_augmentation_contracts():
      metric = (-LogLoss()).with_alias("neg_ll")
      assert metric.resolved_name() == "neg_ll"
      assert metric.metric_name == "neg_ll"

      spec = EvalSpec(primary=Auc(), metrics=[RevenueLift()])
      assert spec.required_columns() == ("amount",)
      assert spec.required_augmentations() == ("risk_weight",)

      evaluated = spec.evaluate(_frame(), pd.Series([0.1, 0.2, 0.8, 0.9]), "target")
      revenue_record = next(record for record in evaluated["metrics"] if record["name"] == "revenue_lift")
      assert revenue_record["augmentations"] == ["risk_weight"]


  def test_scalar_metric_records_excludes_non_scalar_values():
      result = {
          "primary": "auc",
          "metrics": [
              {"name": "auc", "value": 0.8},
              {"name": "threshold_sweep", "value": [{"threshold": 0.5, "precision": 1.0}]},
          ],
      }
      assert is_scalar_value(0.8)
      assert not is_scalar_value(float("nan"))
      assert scalar_metric_records(result) == {"auc": 0.8, "primary": 0.8}
  ```

  Run:
  ```bash
  uv run pytest tests/unit/eval/test_metrics_breadth.py -v
  ```
  Expected: FAIL because `LogLoss`, `ThresholdSweep`, sign/alias helpers,
  `required_augmentations`, and scalar helpers are not implemented.

- [x] Implement `Metric` breadth in `automl/eval/base.py`.

  Required behavior:
  - Keep `Metric.compute(df, y_pred, target_col)` as the abstract extension point.
  - Add `required_columns` as a tuple of strings defaulting to `()` and
    `required_augmentations` as a tuple of strings defaulting to `()`.
  - Add `__neg__()` and `with_alias(alias)` using `copy.copy`.
  - Add `resolved_name()` and keep `metric_name` as a property returning it so Phase 1/2 call
    sites remain simple while the new resolved-name contract lands.
  - Add `Metric.evaluate(df, y_pred, target_col) -> {"name": resolved_name, "value": jsonable_value}` and include
    sign handling only for finite scalar values.
  - Add `is_scalar_value(value)` and `scalar_metric_records(result)`.

- [x] Implement `EvalSpec` breadth in `automl/eval/base.py`.

  Required behavior:
  - Accept primary/metrics as `Metric` or one-item alias mappings such as
    `{"custom_auc": Auc()}`.
  - Expose `metrics` as the locked metric tuple including the primary first.
  - Preserve duplicate-name validation after alias/sign resolution.
  - Return required columns and augmentations in stable first-seen order.
  - Support `evaluate(df, y_pred, target_col, augmentation_frames=None, hash_key=None)` by hash-key joining
    required augmentation frames before metric computation.
  - Return report shape:
    `{"primary": <primary_metric_name>, "metrics": [{"name", "value", "augmentations"}]}`.
  - Require the primary metric value to be finite scalar.

- [x] Implement `LogLoss` and `ThresholdSweep` in `automl/eval/metrics.py`.

  Required behavior:
  - `Auc` keeps `name = "auc"`.
  - `LogLoss` uses `sklearn.metrics.log_loss` and `name = "log_loss"`.
  - `ThresholdSweep(thresholds=[0.25, 0.5])` requires at least one threshold, stores float thresholds,
    and returns a list of `{threshold, precision, recall}` dicts with `zero_division=0`.

- [x] Export `LogLoss` and `ThresholdSweep` from `automl/eval/__init__.py`.

- [x] Update `tests/unit/eval/test_eval_thin_path.py` expectations from the Phase 1 mapping
  shape to the report-record shape. Keep assertions that `EvalResult.to_dict()` omits
  runtime-only `cached`.

- [x] Run eval metric tests.

  Run:
  ```bash
  uv run pytest tests/unit/eval/test_metrics_breadth.py tests/unit/eval/test_eval_thin_path.py -v
  ```
  Expected: PASS.

**Acceptance:** Metric breadth required by A3.1 is covered at unit level.

---

## P3.2 - EvalResult, EvalIndex, And Predictions Schemas

**Files:**
- Modify: `automl/eval/results.py`
- Modify: `automl/eval/__init__.py`
- Test: `tests/unit/eval/test_results_schemas.py`
- Test: `tests/unit/eval/test_eval_thin_path.py`

**Specs:** `spec/07-eval.md` Q6; `spec/02-mlflow-seam.md` section 8 and section 10.

**Legacy source:**
- `automl_legacy/eval/evaluate.py` (`_TOC_ENTRY_KEYS`, report shape)
- `automl_legacy/mlflow/artifacts/predictions.py`

**Migration rows:** `eval/results.py::{EvalResult,EvalIndex,Predictions}`.

**Steps:**
- [x] Write failing schema round-trip tests.

  Add `tests/unit/eval/test_results_schemas.py` with tests equivalent to:
  ```python
  import pandas as pd

  from automl.eval import EvalIndex, EvalResult, Predictions
  from automl.eval.results import EvalIndexEntry


  def test_eval_result_uses_report_shape_and_omits_cached():
      result = EvalResult(
          label="external_augmented",
          eval_dataset_id="v1_abcdef12",
          eval_dataset_kind="external",
          predictions_uri="gs://bucket/eval/external_augmented/predictions.parquet",
          predictions_manifest_uri="gs://bucket/eval/external_augmented/predictions.json",
          augmentations_used=({"name": "risk_weight", "hash8": "12345678"},),
          primary="auc",
          metrics=(
              {"name": "auc", "value": 0.91, "augmentations": []},
              {"name": "threshold_sweep", "value": [{"threshold": 0.5, "precision": 1.0}], "augmentations": []},
          ),
          computed_at="2026-05-27T00:00:00+00:00",
          cached=True,
      )
      payload = result.to_dict()
      assert "cached" not in payload
      assert payload["metrics"][0]["name"] == "auc"
      assert EvalResult.from_dict({**payload, "future": "ignored"}).cached is False


  def test_eval_index_round_trips_entries_and_primary_label():
      index = EvalIndex(
          primary_label="external_augmented",
          evaluations=(
              EvalIndexEntry(
                  label="external_augmented",
                  eval_dataset_id="v1_abcdef12",
                  kind="external",
                  report_path="eval/external_augmented/results.json",
                  eval_dataset_manifest_uri="gs://bucket/eval/datasets/v1_abcdef12/manifest.json",
                  predictions_uri="gs://bucket/eval/external_augmented/predictions.parquet",
                  predictions_manifest_uri="gs://bucket/eval/external_augmented/predictions.json",
                  augmentations_used=({"name": "risk_weight", "hash8": "12345678"},),
                  computed_at="2026-05-27T00:00:00+00:00",
              ),
          ),
      )
      restored = EvalIndex.from_dict(index.to_dict())
      assert restored.primary_label == "external_augmented"
      assert restored.evaluations[0].label == "external_augmented"


  def test_predictions_manifest_and_frame_round_trip():
      frame = pd.DataFrame({"row_id": [1, 2], "y_pred": [0.2, 0.8]})
      predictions = Predictions(
          trial_run_id="run-1",
          eval_dataset_id="v1_abcdef12",
          eval_dataset_kind="external",
          label="external_augmented",
          hash_key=("row_id",),
          frame=frame,
          augmentations_used=({"name": "risk_weight", "hash8": "12345678"},),
          written_at="2026-05-27T00:00:00+00:00",
      )
      manifest = predictions.manifest_dict()
      assert manifest["row_count"] == 2
      assert manifest["hash_key"] == ["row_id"]
      restored = Predictions.from_parts(manifest, frame)
      assert restored.frame.equals(frame)
  ```

  Run:
  ```bash
  uv run pytest tests/unit/eval/test_results_schemas.py -v
  ```
  Expected: FAIL because `EvalIndex`, `EvalIndexEntry`, and `Predictions` do not exist and
  `EvalResult.metrics` is still scalar-only.

- [x] Implement schema dataclasses in `automl/eval/results.py`.

  Required behavior:
  - `EvalResult.metrics` is a variable-length tuple of `dict[str, object]`, preserving non-scalar metric
    values.
  - `EvalResult.cached` remains runtime-only: omitted by `to_dict`, default `False` in
    `from_dict`.
  - `EvalIndexEntry` includes exactly:
    `label`, `eval_dataset_id`, `kind`, `report_path`, `eval_dataset_manifest_uri`,
    `predictions_uri`, `predictions_manifest_uri`, `augmentations_used`, `computed_at`.
  - `EvalIndex` includes `primary_label: str | None`, `evaluations`, and `schema_version = 1`.
  - `Predictions` owns the DataFrame plus a JSON sidecar manifest. The DataFrame is runtime
    data and is not returned by `manifest_dict()`.
  - Every `from_dict` / `from_parts` strips unknown keys and normalizes list fields to tuples.

- [x] Export `EvalIndex`, `EvalIndexEntry`, and `Predictions` from `automl/eval/__init__.py`.

- [x] Run schema tests.

  Run:
  ```bash
  uv run pytest tests/unit/eval/test_results_schemas.py tests/unit/eval/test_eval_thin_path.py -v
  ```
  Expected: PASS.

**Acceptance:** Typed eval result, index, and prediction schemas required by A3.2 exist.

---

## P3.3 - MLflow Trial Eval And Predictions Artifacts

**Files:**
- Create: `automl/mlflow/trial/artifacts/predictions.py`
- Modify: `automl/mlflow/tags.py`
- Modify: `automl/mlflow/trial/artifacts/eval.py`
- Modify: `automl/mlflow/trial/artifacts/__init__.py`
- Modify: `automl/mlflow/trial/artifacts/model.py`
- Modify: `automl/mlflow/trial/artifacts/data.py` if a shared JSON/bytes reader avoids
  duplication.
- Test: `tests/unit/mlflow/test_eval_predictions_artifacts.py`
- Test: `tests/unit/mlflow/test_trial_artifacts.py`

**Specs:** `spec/02-mlflow-seam.md` sections 3.5, 6.3.4, 7, 8, 10; `spec/07-eval.md` Q6.

**Legacy source:**
- `automl_legacy/mlflow/artifacts/eval.py`
- `automl_legacy/mlflow/artifacts/predictions.py`
- `automl_legacy/eval/evaluate.py` (`_read_eval_report`, `_write_eval_artifacts_to_mlflow`)

**Migration rows:** `mlflow/artifacts/eval.py::{validate_eval_label,write_evaluation_results}`;
`mlflow/artifacts/predictions.py::{PredictionsArtifact,write_predictions_gcs}`.

**Steps:**
- [x] Write failing MLflow artifact tests.

  Add `tests/unit/mlflow/test_eval_predictions_artifacts.py` with tests that assert:
  - `write_eval(run_id, label, eval_result)` writes `eval/<label>/results.json` and sets
    `automl.trial.eval.<label>.uri` and
    `automl.trial.eval.<label>.eval_dataset_id`.
  - `load_eval(run_id, label)` returns `EvalResult`.
  - `list_eval(run_id)` returns sorted `(label, eval_dataset_id)` pairs using tags only.
  - `write_eval_index(run_id, eval_index)` writes `eval/manifest.json`.
  - `load_eval_index(run_id)` returns `EvalIndex`, and returns an empty index when absent.
  - `write_predictions(run_id, label, predictions)` writes
    `eval/<label>/predictions.parquet` and `eval/<label>/predictions.json`, sets prediction URI
    tags, and returns a ref with both URIs.
  - `load_predictions(run_id, label)` returns `Predictions` with the original frame.
  - invalid labels reject `".."`, `""`, and labels containing slashes.

  Run:
  ```bash
  uv run pytest tests/unit/mlflow/test_eval_predictions_artifacts.py -v
  ```
  Expected: FAIL because loaders, list functions, index writers, and predictions writer do not
  exist.

- [x] Add canonical prediction tag helpers in `automl/mlflow/tags.py`.

  Required helpers:
  - `eval_predictions_uri(label) -> str`
  - `eval_predictions_manifest_uri(label) -> str`

- [x] Expand `automl/mlflow/trial/artifacts/eval.py`.

  Required behavior:
  - Keep the existing label validation rules and export it as `validate_eval_label`.
  - `write_eval(run_id, label, payload)` writes `eval/<label>/results.json`.
  - `load_eval(run_id, label)` reads the URI tag and deserializes `EvalResult.from_dict`.
  - `list_eval(run_id)` reads run tags and returns sorted `(label, eval_dataset_id)` pairs.
  - `write_eval_index(run_id, payload)` writes `eval/manifest.json`.
  - `load_eval_index(run_id)` reads `eval/manifest.json`; if no tag/artifact exists, return
    `EvalIndex(primary_label=None, evaluations=())`.

- [x] Implement `automl/mlflow/trial/artifacts/predictions.py`.

  Required behavior:
  - Convert `Predictions.frame` to parquet bytes using pandas/pyarrow.
  - Write parquet first and manifest second. If the manifest write fails after a GCS parquet
    write, delete the parquet object when GCS delete support is available; otherwise raise
    `StorageError` with the orphan URI in the message for later cleanup.
  - For local MLflow artifact stores with no GCS bucket, log both files under `eval/<label>/`.
  - `load_predictions` supports both `gs://` and `runs:/<run_id>/<artifact_path>` URIs.
  - `list_predictions(run_id)` returns labels from prediction URI tags.

- [x] Add `load_model(run_id)` to `automl/mlflow/trial/artifacts/model.py`.

  Required behavior:
  - Read the existing `automl.trial.model.uri` tag.
  - Support `gs://` by `gcs.read_bytes(uri)` and `runs:/<run_id>/<artifact_path>` by
    `client.raw().download_artifacts`.
  - `cloudpickle.loads(model_bytes)` and return the model object.
  - This is a seam-local artifact loader for `evaluate()`; do not add `trial/show.py` or a
    trial-domain `load_model` API in Phase 3.

- [x] Export new artifact functions from `automl/mlflow/trial/artifacts/__init__.py`.

- [x] Run MLflow artifact tests.

  Run:
  ```bash
  uv run pytest tests/unit/mlflow/test_eval_predictions_artifacts.py tests/unit/mlflow/test_trial_artifacts.py -v
  ```
  Expected: PASS.

**Acceptance:** MLflow seam can persist and reload multi-instance eval, predictions, and index
artifacts required by A3.2.

---

## P3.4 - Durable EvalDataset For Split View And External Frames

**Files:**
- Modify: `automl/eval/eval_dataset.py`
- Modify: `automl/eval/prepare.py`
- Modify: `automl/eval/_load.py`
- Modify: `automl/eval/__init__.py`
- Modify: `automl/utils/io/gcs.py`
- Test: `tests/unit/eval/test_eval_dataset_identity.py`
- Test: `tests/integration/eval/test_eval_dataset_persistence.py`

**Specs:** `spec/07-eval.md` Q1-Q3, Q6; `spec/05-data.md` Q3-Q4; `spec/00-structural-design.md`
section 8.4.

**Legacy source:**
- `automl_legacy/eval/snapshot.py`
- `automl_legacy/eval/publish.py`
- `automl_legacy/eval/loading.py`

**Migration rows:** `eval/eval_dataset.py` identity/constants/validators; `eval/_load.py`
`LoadedEvalDataset`; `eval/prepare.py::prepare_eval_dataset`; `eval/prepare.py`
GCS helper imports.

**Steps:**
- [x] Write failing unit tests for EvalDataset identity.

  Add `tests/unit/eval/test_eval_dataset_identity.py` with tests that assert:
  - `compute_eval_dataset_identity(kind="split_view", of_dataset_id="v1_data", split_id_col="SPLITID", buckets=((80, 90), (95, 100)), target_column="target", hash_key=("row_id",))`
    is deterministic and does not require a DataFrame.
  - Two different split-view bucket recipes that realize the same rows still produce different
    ids when the recipe differs.
  - `compute_eval_dataset_identity(kind="external", df=frame, target_column="target", hash_key=("row_id",))`
    changes when frame content changes.
  - `EvalDataset.to_dict()` / `from_dict()` round-trip route fields and URI properties.
  - Split-view manifests do not carry realized `schema_hash` or `content_hash`.
  - External manifests carry `schema_hash`, `content_hash`, and `data_gcs_uri`.
  - Bad bucket ranges, missing target, missing hash key, and duplicate external hash keys raise
    with messages naming the bad field.

  Run:
  ```bash
  uv run pytest tests/unit/eval/test_eval_dataset_identity.py -v
  ```
  Expected: FAIL because the current `EvalDataset` is process-local and split-name-only.

- [x] Implement `EvalDataset`, `LoadedEvalDataset`, and identity helpers in
  `automl/eval/eval_dataset.py`.

  Required behavior:
  - `EvalDataset.kind` is `"split_view"` or `"external"`.
  - Route fields are snapshotted from `mlflow.client.bound()` / active session:
    `gcs_bucket`, `gcs_prefix`, `project_name`, `experiment_id`, `dry_run`, `namespace`.
  - `manifest_gcs_uri` derives as
    `gs://<bucket>/<route_prefix>/eval/datasets/<eval_dataset_id>/manifest.json`.
  - `data_gcs_uri` is the matching `data.parquet` for external datasets and `None` for
    split-view datasets.
  - Split-view identity hashes only the recipe:
    kind, target column, hash key, of dataset id, split id column, and normalized buckets.
  - External identity hashes content and schema.
  - `from_dict` strips unknown fields.

- [x] Replace the process-local `_RECIPES` registry in `automl/eval/prepare.py`.

  Required split-view behavior:
  - Keep the runner-friendly entry point:
    `prepare_eval_dataset(session=active, dataset_id=loaded_fit.id, split=run_config.eval_split)`.
  - Resolve `split` to buckets through `active.config.require_run_config().splits`.
  - Resolve the parent `Dataset` from `data.list_datasets(session=active)` and read its manifest.
  - Write only the split-view manifest. Do not realize or hash the frame at publish time.
  - Return `(EvalDataset, cached)` where `cached` is runtime-only and is `True` only when the
    existing manifest was reused without rewriting.

  Required external behavior:
  - Add `prepare_eval_dataset(session=active, frame=external_df, kind="external", target_col=target_col, hash_key=hash_key, provenance=provenance, overwrite=False)`.
  - Validate target column, hash-key presence, and hash-key uniqueness.
  - Write `data.parquet` and `manifest.json` under `eval/datasets/<eval_dataset_id>/`.
  - If both objects already exist and `overwrite=False`, validate the manifest and return the
    same `EvalDataset` with `cached=True`.
  - If both objects already exist and `overwrite=True`, rewrite both objects and return
    `cached=False`.
  - If exactly one object exists, raise `StorageError` naming the partial object set.

- [x] Update `automl/eval/_load.py`.

  Required behavior:
  - `load_eval_dataset(eval_dataset_id, session=active)` reads the durable manifest.
  - For split-view, call `data.load_dataset_by_id(of_dataset_id, split_range=buckets, session=active)`.
  - For external, read the parquet frame from `EvalDataset.data_gcs_uri`.
  - Return `LoadedEvalDataset(df, dataset, target_column, hash_key, row_ids)` where `row_ids`
    contains the hash-key columns in eval row order.
  - Empty split-view slices raise at load/evaluate time, matching spec 07 Q2.

- [-] Add `utils.io.gcs.delete(uri)` only if P3.3's partial-write rollback needs it.

  Not needed in Phase 3; partial-write handling is fail-fast/rewrite via existing overwrite paths.

  Required behavior:
  - Parse the URI with `parse_gcs_uri`.
  - Call `bucket.blob(name).delete()`.
  - Missing objects are treated as already deleted.

- [x] Write failing integration tests with fake GCS.

  Add `tests/integration/eval/test_eval_dataset_persistence.py` to prove:
  - Split-view `prepare_eval_dataset` writes only a manifest and `load_eval_dataset` delegates
    to `data.load_dataset_by_id(dataset_id, split_range=((80, 100),))`.
  - External `prepare_eval_dataset` writes both parquet and manifest, returns the same id with
    `cached=True` on a second call, rewrites when `overwrite=True`, and `load_eval_dataset`
    returns the original rows.

- [x] Run eval dataset tests.

  Run:
  ```bash
  uv run pytest tests/unit/eval/test_eval_dataset_identity.py tests/integration/eval/test_eval_dataset_persistence.py tests/unit/eval/test_eval_thin_path.py -v
  ```
  Expected: PASS.

**Acceptance:** Durable EvalDataset identity and load path required by A3.2 are covered.

---

## P3.5 - Augmentation Publish, Load, And EvalSpec Join

**Files:**
- Modify: `automl/eval/eval_dataset.py`
- Modify: `automl/eval/prepare.py`
- Modify: `automl/eval/_load.py`
- Modify: `automl/utils/io/gcs.py`
- Test: `tests/unit/eval/test_augmentations.py`
- Test: `tests/integration/eval/test_augmentation_integration.py`

**Specs:** `spec/07-eval.md` Q6 and "Remaining surface"; `spec/00-structural-design.md`
section 8.4.

**Legacy source:**
- `automl_legacy/eval/snapshot.py`
- `automl_legacy/eval/publish.py`
- `automl_legacy/eval/evaluate.py` augmentation loading helpers.

**Migration rows:** `eval/eval_dataset.py::Augmentation`; `eval/prepare.py::prepare_eval_augmentation`;
augmentation manifest validators.

**Steps:**
- [x] Write failing unit tests for augmentation identity and validation.

  Add `tests/unit/eval/test_augmentations.py` with tests that assert:
  - `compute_augmentation_identity(eval_dataset_id, name, frame, hash_key)` is deterministic
    and includes content hash.
  - Augmentation names must start with lowercase and contain lowercase letters, numbers, or
    underscores.
  - The frame must include all hash-key columns, unique hash-key rows, and at least one
    non-hash-key augmentation column.
  - `Augmentation.to_dict()` / `from_dict()` round-trip path properties and route fields.

  Run:
  ```bash
  uv run pytest tests/unit/eval/test_augmentations.py -v
  ```
  Expected: FAIL because `Augmentation` does not exist.

- [x] Implement `Augmentation` and helper functions in `automl/eval/eval_dataset.py`.

  Required behavior:
  - `Augmentation` includes `eval_dataset_id`, `name`, `hash8`, `content_hash`, `hash_key`,
    route fields, and `schema_version = 1`.
  - `data_gcs_uri` derives under
    `eval/datasets/<eval_dataset_id>/augmentations/<name>__<hash8>/data.parquet`.
  - `manifest_gcs_uri` derives beside it.

- [x] Add GCS prefix listing helpers in `automl/utils/io/gcs.py`.

  Required behavior:
  - `list_blob_names(uri_or_bucket, prefix=None)` returns sorted blob names.
  - `list_prefixes(uri_or_bucket, prefix=None)` returns sorted prefix strings with trailing
    slash.
  - Keep the helpers generic; no AutoML routing logic belongs in `utils`.

- [x] Implement `prepare_eval_augmentation(session=active, eval_dataset_id=eval_dataset.id, frame=augmentation_frame, name="risk_weight", overwrite=False)` in `automl/eval/prepare.py`.

  Required behavior:
  - Load the base eval dataset manifest to get hash key and target columns.
  - Load the base eval frame through `load_eval_dataset` for row-id and column-overlap checks.
  - Reject augmentation columns that overlap base eval columns, excluding hash-key columns.
  - Reject row ids not present in the base eval dataset.
  - Reject column overlap with existing augmentations on the same eval dataset.
  - Write parquet + manifest and return `(Augmentation, cached)`.
  - Return the cached augmentation with `cached=True` when both objects exist, `overwrite=False`,
    and the manifest matches.
  - Rewrite both objects and return `cached=False` when `overwrite=True`.

- [x] Add augmentation loading helpers used by `evaluate()`.

  Required behavior:
  - Given `eval_dataset_id` and required augmentation names, find matching
    `<name>__<hash8>` folders under that eval dataset.
  - Pick the newest `created_at` for a repeated name.
  - Return both frames for metric joins and usage records:
    `{name, hash8, data_uri, manifest_uri}`.
  - Raise `ValueError("augmentations not published on eval dataset: ['risk_weight']")` for missing
    required names.

- [x] Write integration tests for augmentation join through `EvalSpec.evaluate`.

  Add `tests/integration/eval/test_augmentation_integration.py` with:
  - A base external eval dataset.
  - A `risk_weight` augmentation keyed by the same hash key.
  - A metric declaring `required_augmentations = ("risk_weight",)` and
    `required_columns = ("risk_weight",)`.
  - `EvalSpec.evaluate(df, y_pred, "target", augmentation_frames={"risk_weight": frame}, hash_key=("row_id",))`
    computes the metric and records `augmentations == ["risk_weight"]`.
  - A second `prepare_eval_augmentation` call with the same eval dataset, frame, name, and
    `overwrite=False` returns the same id with `cached=True`; `overwrite=True` rewrites and
    returns `cached=False`.

- [x] Run augmentation tests.

  Run:
  ```bash
  uv run pytest tests/unit/eval/test_augmentations.py tests/integration/eval/test_augmentation_integration.py tests/unit/eval/test_metrics_breadth.py -v
  ```
  Expected: PASS.

**Acceptance:** Augmentation contract and metric integration required by A3.2 are covered.

---

## P3.6 - Stateful evaluate() Owns Persistence And Scalar Logging

**Files:**
- Modify: `automl/eval/evaluate.py`
- Modify: `automl/eval/_load.py`
- Modify: `automl/mlflow/trial/artifacts/__init__.py`
- Test: `tests/integration/eval/test_evaluate_persistence.py`
- Test: `tests/unit/eval/test_eval_thin_path.py`

**Specs:** `spec/07-eval.md` Q5-Q6; `spec/02-mlflow-seam.md` sections 6.3.4 and 10;
`spec/08-runner.md` Q5.

**Legacy source:**
- `automl_legacy/eval/evaluate.py`
- `automl_legacy/eval/compatibility.py`
- `automl_legacy/mlflow/artifacts/predictions.py`

**Migration rows:** `eval/evaluate.py::evaluate`; `eval/runner.py::run -> evaluate_frame`;
out-of-domain caller `runner/_execute.py` consumes `EvalResult`.

**Steps:**
- [x] Write failing integration tests for `evaluate()` persistence.

  Add `tests/integration/eval/test_evaluate_persistence.py` with tests that assert:
  - `evaluate(session=active, model_run_id=run_id, eval_dataset_id=eval_dataset.id, label="holdout", set_as_primary_label=True, _model=model, overwrite=False)` writes
    predictions, eval result, EvalIndex, scalar metrics, and eval tags.
  - `run.data.metrics` includes `holdout.auc` and `holdout.negative_log_loss`.
  - `run.data.metrics` does not include `holdout.threshold_sweep`.
  - The bare primary metric key is logged only for the primary label.
  - `mlflow.trial.artifacts.load_eval(run_id, "holdout")` and
    `load_predictions(run_id, "holdout")` round-trip.
  - A second label for the same run updates EvalIndex without replacing the first label.
  - Calling `evaluate(session=active, model_run_id=run_id, eval_dataset_id=eval_dataset.id, label="holdout", _model=None)` loads the model through
    `mlflow.trial.artifacts.load_model(run_id)` and does not import `mlflow` in eval code.
  - Calling `evaluate` again with the same run id, eval dataset id, label, and `overwrite=False`
    after complete artifacts exist returns `cached=True` without rewriting predictions/results.
  - Calling `evaluate` again with the same run id, eval dataset id, label, and `overwrite=True`
    recomputes and returns `cached=False`.

  Run:
  ```bash
  uv run pytest tests/integration/eval/test_evaluate_persistence.py -v
  ```
  Expected: FAIL because `evaluate()` currently returns an in-memory result and does not write
  predictions, EvalIndex, or scalar metrics.

- [x] Update prediction flow in `automl/eval/evaluate.py`.

  Required behavior:
  - Resolve session through the standard `session` convention.
  - Load eval dataset via `load_eval_dataset`.
  - If `_model is None`, call `mlflow.trial.artifacts.load_model(model_run_id)`.
  - Predict on the eval frame with the target column dropped.
  - Build `Predictions` with hash-key row ids and `y_pred`.
  - Write predictions through the MLflow seam before writing `EvalResult`.

- [x] Update metric flow in `automl/eval/evaluate.py`.

  Required behavior:
  - Use the explicit `eval_spec` argument when supplied; otherwise use
    `active.config.require_eval_spec()`.
  - Load required augmentations by metric names before metric computation.
  - Compute the full locked metric set, including non-scalar records.
  - Build one `EvalResult` with the report-record metric shape.
  - Before recomputing, when `overwrite=False`, attempt to load existing complete artifacts for
    the label and return them with runtime-only `cached=True`.
  - If any required artifact is missing, partial, invalid, or stale relative to the requested
    eval dataset/metric spec, recompute and return runtime-only `cached=False`.

- [x] Move eval artifact/index/scalar persistence into `evaluate()`.

  Required behavior:
  - Write predictions, then `EvalResult`, then `EvalIndex`.
  - Update or replace the index entry for the label while preserving other labels.
  - Set `primary_label` when `set_as_primary_label=True`.
  - Log every finite scalar metric under `<label>.<metric>`.
  - If this label is the primary label, log the bare primary metric name as a per-trial
    convenience and set an `eval.primary_metric` tag.
  - Return the same `EvalResult` that was persisted.

- [x] Keep `evaluate_frame(y_pred=y_pred, df=df, spec=spec, target_col="target")` pure.

  Required behavior:
  - No MLflow or GCS writes.
  - Accept `spec` and `target_col`.
  - Return the same report shape as `EvalSpec.evaluate`.

- [x] Run evaluate persistence tests.

  Run:
  ```bash
  uv run pytest tests/integration/eval/test_evaluate_persistence.py tests/unit/eval/test_eval_thin_path.py -v
  ```
  Expected: PASS.

**Acceptance:** A3.1 scalar logging and A3.2 Predictions/EvalIndex persistence are covered at
integration level.

---

## P3.7 - Runner Integration And Phase 2 Regression Guard

**Files:**
- Modify: `automl/runner/trial.py`
- Test: `tests/integration/runner/test_one_trial_local.py`
- Test: `tests/integration/runner/test_phase3_eval_runner.py`

**Specs:** `spec/08-runner.md` Q5; `spec/07-eval.md` Q5-Q6.

**Legacy source:**
- `automl_legacy/runner/_execute.py`
- `automl_legacy/runner/_stages.py`

**Migration rows:** `runner/_execute.py::run_trial` eval-consumption row.

**Steps:**
- [x] Write failing runner integration tests for evaluate-owned persistence.

  Add `tests/integration/runner/test_phase3_eval_runner.py` with tests that assert:
  - The runner still loads only the fit slice directly.
  - Eval loads happen inside `automl.eval._load.load_eval_dataset`.
  - Trial `test` and best-effort `train` labels both appear in `artifacts.list_eval(run_id)`.
  - `artifacts.load_eval_index(run_id).evaluations` contains both labels.
  - `artifacts.load_predictions(run_id, "test").frame` has one row per eval slice row.
  - The Phase 2 WOE required-transformer model still passes.

  Run:
  ```bash
  uv run pytest tests/integration/runner/test_phase3_eval_runner.py -v
  ```
  Expected: FAIL because runner still owns `_log_eval` and no predictions/index exist.

- [x] Simplify `automl/runner/trial.py` eval logging.

  Required behavior:
  - Keep `prepare_eval_dataset(session=active, dataset_id=loaded_fit.id, split=run_config.eval_split)` for split-view eval, now receiving an `EvalDataset`
    plus runtime `cached` flag and passing `eval_dataset.id` to `evaluate()`.
  - Remove runner-owned `artifacts.write_eval` and scalar metric logging. `evaluate()` owns
    those writes.
  - Keep best-effort train eval as a separate label with `set_as_primary_label=False`.
  - Convert returned `EvalResult` to `TrialResult.metrics` using `scalar_metric_records(eval_result.to_dict())`
    with the `"primary"` helper removed.

- [x] Update `tests/integration/runner/test_one_trial_local.py`.

  Required assertion changes:
  - Continue checking `test.auc`, bare `auc`, and `train.auc` metrics.
  - Continue checking eval URI tags for `test` and `train`.
  - Add checks for predictions URI tags and EvalIndex.
  - Update persisted eval result assertions to the report-record metric shape.

- [x] Run runner and Phase 2 integration tests.

  Run:
  ```bash
  uv run pytest tests/integration/runner/test_one_trial_local.py tests/integration/runner/test_phase3_eval_runner.py tests/integration/homecredit -v
  ```
  Expected: PASS.

- [x] Run full non-external suite.

  Run:
  ```bash
  uv run pytest tests/unit tests/contracts tests/integration -v
  ```
  Expected: PASS.

**Acceptance:** Runner integration still satisfies the Phase 2 WOE path and now produces
Phase 3 eval/prediction/index artifacts.

---

## P3.8 - Phase 3 External Eval E2E Gate

**Files:**
- Create: `tests/e2e/test_phase3_eval_breadth.py`
- Modify: no production files unless this gate exposes a bug.

**Specs:** `plan/acceptance-checklist.md` A3.1-A3.2; `spec/07-eval.md`; `spec/02-mlflow-seam.md`.

**Steps:**
- [x] Write the gated external e2e test.

  Add `tests/e2e/test_phase3_eval_breadth.py` guarded by:
  ```python
  @pytest.mark.skipif(
      not (
          os.environ.get("AUTOML_PHASE3_E2E")
          and os.environ.get("GCS_BUCKET")
          and os.environ.get("MLFLOW_TRACKING_URI")
          and os.environ.get("GCP_PROJECT")
      ),
      reason=(
          "Phase 3 e2e requires AUTOML_PHASE3_E2E, GCS_BUCKET, "
          "GCP_PROJECT, and MLFLOW_TRACKING_URI"
      ),
  )
  ```

  Test scenario:
  - `use_project("example_homecredit", repo_root=repo_root)`.
  - `materialize(session=active)`.
  - `run_trial("example_homecredit", session=active)` and assert `FINISHED`.
  - Load the normal eval slice with `load_dataset_by_id(loaded.id, split_name="test")`.
  - Build an external eval frame from that slice with stable hash-key columns and target.
  - `external, external_cached = prepare_eval_dataset(kind="external", frame=external_frame, target_col=active.config.target_column, hash_key=loaded.dataset.hash_key, provenance={"source": "phase3_e2e"})`.
  - Build a `risk_weight` augmentation frame with the same hash key and one numeric
    augmentation column.
  - `augmentation, augmentation_cached = prepare_eval_augmentation(eval_dataset_id=external.id, frame=augmentation_frame, name="risk_weight")`.
  - Define a local test metric:
    ```python
    class WeightedMeanScore(Metric):
        name = "weighted_mean_score"
        required_columns = ("risk_weight",)
        required_augmentations = ("risk_weight",)

        def compute(self, df, y_pred, target_col):
            del target_col
            return float((df["risk_weight"] * y_pred).mean())
    ```
  - Call:
    ```python
    external_result = evaluate(
        session=active,
        model_run_id=result.run_id,
        eval_dataset_id=external.id,
        eval_spec=EvalSpec(
            primary=Auc(),
            metrics=[
                -LogLoss(),
                ThresholdSweep(thresholds=[0.3, 0.5, 0.7]),
                WeightedMeanScore(),
            ],
        ),
        label="external_augmented",
        set_as_primary_label=True,
    )
    ```
  - Assert `external_result.eval_dataset_kind == "external"`.
  - Assert persisted `EvalResult`, `Predictions`, and `EvalIndex` round-trip through
    `automl.mlflow.trial.artifacts`.
  - Assert `artifacts.list_eval(result.run_id)` includes `("external_augmented", external.id)`.
  - Assert MLflow run metrics include
    `external_augmented.auc`, `external_augmented.negative_log_loss`,
    `external_augmented.weighted_mean_score`, and bare `auc`.
  - Assert the persisted metric records include `threshold_sweep`.
  - Assert all prediction/eval dataset/augmentation URIs are `gs://` and exist in GCS.

  Run without the env gate:
  ```bash
  uv run pytest tests/e2e/test_phase3_eval_breadth.py -v
  ```
  Expected: SKIPPED with the Phase 3 env reason.

- [x] Run the external Phase 3 gate against the original MLflow server.

  Load this worktree's `.env`, then run:
  ```bash
  AUTOML_PHASE3_E2E=1 MLFLOW_TRACKING_URI=http://127.0.0.1:54321 uv run pytest tests/e2e/test_phase3_eval_breadth.py -v
  ```
  Expected: PASS. The run is visible under experiment
  `example_homecredit/example-homecredit`. Sparse trial artifact tabs are not failure when GCS is
  configured; verify URI tags and GCS objects.

- [x] Run the full suite after the external gate.

  Run:
  ```bash
  uv run pytest tests/unit tests/contracts tests/integration -v
  ```
  Expected: PASS.

**Acceptance:** A3.1 and A3.2 have external evidence.

---

## P3.9 - Docs Closeout

**Files:**
- Modify: `docs/superpowers/automl-refactor/README.md`
- Modify: `docs/superpowers/automl-refactor/plan/README.md`
- Modify: `docs/superpowers/automl-refactor/plan/implementation-strategy.md` only if phase
  boundaries changed.
- Modify: `docs/superpowers/automl-refactor/plan/acceptance-checklist.md`
- Modify: `docs/superpowers/automl-refactor/plan/migration-checklist.md`
- Modify: `docs/superpowers/automl-refactor/plan/phases/phase-3-eval-breadth.md`

**Specs:** execution docs are status authority; specs change only if running evidence proves a
design correction is needed.

**Steps:**
- [x] Update `acceptance-checklist.md`.

  Required updates:
  - Mark A3.1 `[x]` only after P3.8 passes.
  - Mark A3.2 `[x]` only after P3.8 passes.
  - Record exact command evidence and date.

- [x] Update `migration-checklist.md`.

  Required updates:
  - Flip Phase 3-covered eval, MLflow eval/predictions, and runner eval-consumption rows to
    `[x]`.
  - Keep Phase 4+ rows unchecked.
  - Do not mark CLI, experiment/trial views, cleanup, agent, or full trial read rows complete.

- [x] Update front-door status docs.

  Required updates:
  - `docs/superpowers/automl-refactor/README.md`: Phase 3 complete; next action is Phase 4
    detailed plan.
  - `plan/README.md`: same status and evidence.
  - `implementation-strategy.md`: only update if implementation changed the Phase 4 boundary.

- [x] Update this phase plan's review checkpoint.

  Required updates:
  - Mark completed task checkboxes.
  - Add a closeout note with final evidence.
  - List any consciously deferred eval work for Phase 4+.

- [x] Run stale-status scans.

  Run:
  ```bash
  rg -n "Phase 3.*not started|NEXT ACTION.*Phase 3|A3\\.[12].*\\[ \\]|eval snapshot|eval_snapshot" docs/superpowers/automl-refactor automl tests projects/example_homecredit
  ```
  Expected: no stale status claims. Any remaining `eval_snapshot` hits must be legacy notebooks
  or explicitly deferred docs; update new code/tests/docs to `eval_dataset`.

  Result: no stale Phase 3 status claims. Remaining `eval_snapshot` hits are historical rename
  notes in specs/migration rows or old `projects/example_homecredit` notebooks deferred with the
  later CLI/notebook surface work.

- [x] Run final verification.

  Run:
  ```bash
  uv run pytest tests/unit tests/contracts tests/integration -v
  AUTOML_PHASE3_E2E=1 MLFLOW_TRACKING_URI=http://127.0.0.1:54321 uv run pytest tests/e2e/test_phase3_eval_breadth.py -v
  uv run pytest tests/contracts -v
  ```
  Expected: PASS.

**Acceptance:** Phase 3 can be handed off with docs and checklists consistent.
