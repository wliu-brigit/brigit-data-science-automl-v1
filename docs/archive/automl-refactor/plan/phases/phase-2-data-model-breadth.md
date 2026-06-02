# Phase 2 Data And Model Breadth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development`
> (recommended) or `superpowers:executing-plans` to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** thicken the proven Phase 1 Home Credit path just enough to pass A2.1-A2.5:
source/index breadth, FeatureRegistry lineage, L1-L4 data integrity with multi-range loads,
project-mandated required transformers, and project-level profile artifacts.

**Architecture:** keep the Phase 1 straight-line runner green while broadening only the
contracts that Phase 2 gates exercise. Data remains split-at-load and owns Dataset identity;
model owns the required-preprocessing contract and gate; MLflow remains the only package that
imports PyPI `mlflow`. Project-overview work is limited to the profile artifact path required by
A2.5, not experiment views or trial read models.

**Tech Stack:** Python 3.11 via `uv`; pytest; pandas/numpy/pyarrow; scikit-learn
`ColumnTransformer`; cloudpickle; original local MLflow at `http://127.0.0.1:54321` with the
local `.env` loaded in this refactor worktree; GCS bucket
`gs://automl-homecredit-kaggle-wliu`.

**Acceptance:** `plan/acceptance-checklist.md` rows **A2.1-A2.5**.

---

## Review Checkpoint

Phase 2 is complete as of 2026-05-27. This file is the execution record for the gate; the next
action is to write and review the Phase 3 detailed plan before implementation. Phase 1/2 paths
must stay green after every later task.

## Non-Negotiable Decisions

- New code must never import `automl_legacy`.
- Only `automl/mlflow/**` may import the PyPI `mlflow` package.
- Use `uv` for all commands.
- Write failing tests before implementation in every task.
- Use the original MLflow server for external gates:
  `MLFLOW_TRACKING_URI=http://127.0.0.1:54321` after loading this worktree's `.env`.
- Keep Phase 2 thin. Do not add eval metrics breadth, external eval, predictions, experiment
  views, trial show/load, cleanup, agent loop, full CLI breadth, or namespace/dry-run breadth
  unless an A2 gate proves it is required.
- `load_dataset_by_trial()` resolves named splits from the trial contract, not current
  `config.py`.
- Required-transformer enforcement lives in `model/checks.py` and is invoked by
  `validate.model`; project code declares data, it does not enforce.

## Task DAG

```
P2.0 baseline guard
  -> P2.1 source/index breadth
  -> P2.2 FeatureRegistry breadth
  -> P2.3 L1-L4 validators + multi-range loader
  -> P2.4 RequiredTransformer + Home Credit WOE trial
  -> P2.5 profile + project-overview artifacts
  -> P2.6 Phase 2 e2e gate
  -> P2.7 docs closeout
```

Each task is a review boundary. If specs contradict running evidence, stop and flag the
contradiction before changing code.

---

## P2.0 — Baseline Guard And Scope Ratchet

**Files:**
- Modify: no production files.
- Test/read: existing Phase 1 tests.

**Specs:** `plan/implementation-strategy.md` §5-§6; `plan/acceptance-checklist.md` A1/A2.

**Steps:**
- [x] Run the Phase 1 non-external suite before changing code.

  Run:
  ```bash
  uv run pytest tests/unit tests/contracts tests/integration/data_pipeline tests/integration/homecredit tests/integration/runner -v
  ```
  Expected: PASS. If this fails, fix the regression before starting Phase 2.

- [x] Run the architecture ratchets.

  Run:
  ```bash
  uv run pytest tests/contracts -v
  rg 'automl_legacy' automl projects tests
  rg '(^|\s)(import|from) mlflow' automl projects tests
  ```
  Expected: contract tests pass; `automl_legacy` appears only in ratchet/doc text; PyPI
  `mlflow` imports appear only under `automl/mlflow/` or tests intentionally verifying the seam.

- [x] Leave A2 rows unchecked. This task establishes the starting line only.

**Acceptance:** Phase 1 remains green before Phase 2 changes begin.

---

## P2.1 — Data Source And DatasetIndex Breadth

**Files:**
- Create: `automl/data/sources/gcs_parquet.py`
- Create: `automl/data/sources/snowflake.py`
- Modify: `automl/data/sources/__init__.py`
- Modify: `automl/data/__init__.py`
- Modify: `automl/data/sources/base.py`
- Modify: `automl/data/pipeline.py`
- Modify: `automl/utils/io/gcs.py`
- Test: `tests/unit/data/test_sources_breadth.py`
- Test: `tests/integration/data_pipeline/test_materialize_load.py`

**Specs:** `spec/05-data.md` Q2, Q3, Q4, Q7, §5 exports; `spec/01-project-context.md` §6;
`spec/02-mlflow-seam.md` §6.1/§6.2.3.

**Legacy source:**
- `automl_legacy/data/sources.py`
- `automl_legacy/data/pipeline.py`
- `automl_legacy/io/gcs.py`
- `automl_legacy/io/snowflake.py`

**Migration rows:** `data/sources.py::SnowflakeSource`, `data/sources.py::GCSParquetSource`,
`data/__init__.py`, `io/gcs.py` parquet helpers. Adapter `*Pipeline` wrappers stay `[-]`.

**Steps:**
- [x] Write failing unit tests for concrete source exports and identities.

  Add tests that assert:
  - `from automl.data import LocalCSVSource, GCSParquetSource, SnowflakeSource` works.
  - `GCSParquetSource(gcs_uri="gs://bucket/path/train.parquet", hash_key="row_id").identity()`
    returns `kind="gcs_parquet"`, the URI, and normalized hash-key columns.
  - invalid GCS URIs raise `ValueError` naming the bad URI.
  - `SnowflakeSource(base_table="APP", base_data_sql="base.sql", training_data_sql="train.sql")`
    is accepted by `DataSpec`, exposes `kind="snowflake"`, and its `identity()` includes the
    SQL paths and Snowflake database/schema env values.
  - `SnowflakeSource.load(...)` raises a clear `NotImplementedError` or `StorageError` in this
    Phase 2 stub instead of importing `snowflake` from new data code.

  Run:
  ```bash
  uv run pytest tests/unit/data/test_sources_breadth.py -v
  ```
  Expected: FAIL because `GCSParquetSource` and `SnowflakeSource` do not exist.

- [x] Implement `GCSParquetSource`.

  Behavior:
  - Parse `gs://` URIs via `automl.utils.io.gcs.parse_gcs_uri`.
  - `load(..., nrows=None)` reads with `gcs.read_parquet(uri)` and returns `df.head(nrows)` when
    a row limit is provided.
  - `identity()` returns only deterministic source fields: `kind`, `gcs_uri`, and `hash_key`.
  - Do not add legacy `GCSParquetPipeline`.

- [x] Implement `SnowflakeSource` as the explicit Phase 2 stub.

  Behavior:
  - Dataclass fields: `base_table`, `base_data_sql`, `training_data_sql`.
  - `identity()` includes deterministic env names (`SNOWFLAKE_DATABASE`,
    `SNOWFLAKE_SCHEMA`) and SQL paths.
  - `load()` raises a clear unsupported message for the stub path, because the A2 gate only
    requires resolution, not live Snowflake execution.
  - `artifact_files()` may return `{}` in Phase 2 unless a Snowflake-backed gate is added.

- [x] Add a base optional source-trace hook without wiring project-overview writes.

  Add to `DataSource`:
  ```python
  def artifact_files(self, pipeline: "DataPipeline") -> dict[str, Path]:
      return {}
  ```
  Keep it dormant for Phase 2. Source-trace project-overview logging is not required by A2.1.

- [x] Add `gcs.read_parquet_head(uri, rows)` only if the source implementation needs a helper.

  It may be a small wrapper over `read_parquet(uri).head(rows)` in Phase 2; broad pyarrow row-group
  pushdown remains deferred.

- [x] Expand the data pipeline integration test to prove `build_dataset()` works with
  `GCSParquetSource` using the existing fake GCS client.

  Test shape:
  - Write a tiny DataFrame to fake GCS through `gcs.write_parquet`.
  - Build a `DataSpec(source=GCSParquetSource(..., hash_key="row_id"))`.
  - Call `build_dataset(session=...)`.
  - Assert `dataset.source_identity["kind"] == "gcs_parquet"`, `hash_key == ("row_id",)`,
    `SPLITID` exists, and no GCS materialization objects were written.

- [x] Run source/index tests.

  Run:
  ```bash
  uv run pytest tests/unit/data/test_sources_breadth.py tests/integration/data_pipeline/test_materialize_load.py -v
  ```
  Expected: PASS.

- [x] Run the local Phase 1 runner integration to prove A2.1 did not regress the one-trial path.

  Run:
  ```bash
  uv run pytest tests/integration/runner/test_one_trial_local.py -v
  ```
  Expected: PASS.

**Acceptance:** A2.1 is covered; final external evidence passed in P2.6.

---

## P2.2 — FeatureRegistry Breadth

**Files:**
- Modify: `automl/data/features.py`
- Modify: `automl/data/pipeline.py`
- Test: `tests/unit/data/test_feature_registry_breadth.py`
- Test: `tests/unit/data/test_sources_pipeline_contract.py`

**Specs:** `spec/05-data.md` Q6-Q7; `spec/06-model.md` implementation notes.

**Legacy source:**
- `automl_legacy/core/feature_registry.py`
- `automl_legacy/data/pipeline.py` quality-filter calls

**Migration rows:** `core/feature_registry.py::FeatureEntry`, `FeatureRegistry`.

**Steps:**
- [x] Write failing FeatureRegistry breadth tests.

  Tests must assert:
  - `FeatureEntry` has `derived: bool = False` and `source_columns: tuple[str, ...] = ()`.
  - `FeatureRegistry.add_derived("amt_log", "num", ("amt_credit",), model=True)` creates an
    available, feature, model, derived entry with the exact source columns.
  - `add_derived` raises `ValueError` on duplicate names and `KeyError` on missing sources.
  - `to_dataframe()` / `from_dataframe()` round-trip `derived` and JSON-serialized
    `source_columns`.
  - `get_by_flag("derived")`, `get_by_dtype("num")`, `columns`, `__contains__`, `select()`, and
    `cast(inplace=False)` behave like the stable model-facing contract.
  - Golden/weak learning flags are absent from new registry CSV output.

  Run:
  ```bash
  uv run pytest tests/unit/data/test_feature_registry_breadth.py -v
  ```
  Expected: FAIL on any missing registry methods or serialization behavior.

- [x] Implement only the registry APIs needed by Phase 2 and model code.

  Carry forward from legacy where useful:
  - `columns` property
  - `__len__`, `__contains__`, `__repr__`
  - `get_by_flag`
  - `get_by_dtype`
  - `select`
  - `cast`
  - `add_comment` / `get_comment` if needed by `cast` or tests

  Keep learning APIs deleted: no `golden`, `weak`, `apply_learning_flags`, or
  `import_learning_flags`.

- [x] Ratchet strict constant-drop behavior.

  Add a test with `constant_drop_threshold=1.0` and a strict-constant non-protected column.
  Expected: the column is absent from `LoadedDataset.df`, absent from the materialized registry
  row set, and protected target/hash/metadata columns survive even if constant.

- [x] Run registry and pipeline tests.

  Run:
  ```bash
  uv run pytest tests/unit/data/test_feature_registry_breadth.py tests/unit/data/test_sources_pipeline_contract.py -v
  ```
  Expected: PASS.

- [x] Run the local runner integration.

  Run:
  ```bash
  uv run pytest tests/integration/runner/test_one_trial_local.py -v
  ```
  Expected: PASS.

**Acceptance:** A2.2 is covered; final external evidence passed in P2.6.

---

## P2.3 — Full Data Validators, Multi-Range Loader, And Trial Replay

**Files:**
- Modify: `automl/data/contract.py`
- Modify: `automl/data/registry.py`
- Modify: `automl/mlflow/trial/artifacts/data.py`
- Modify: `automl/mlflow/trial/artifacts/__init__.py`
- Modify: `automl/mlflow/trial/logging.py`
- Modify: `automl/mlflow/trial/__init__.py`
- Modify: `automl/mlflow/experiment/queries.py`
- Test: `tests/unit/data/test_contract_validators.py`
- Test: `tests/integration/data_pipeline/test_trial_replay.py`

**Specs:** `spec/05-data.md` Q3, Q9, §5 validators; `spec/07-eval.md` Q1-Q2;
`spec/08-runner.md` Q4-Q5; `spec/02-mlflow-seam.md` §6.3.4.

**Legacy source:**
- `automl_legacy/data/contract.py`
- `automl_legacy/data/run_snapshot.py`
- `automl_legacy/data/snapshot.py`
- `automl_legacy/runner/_execute.py`

**Migration rows:** `data/contract.py::validate_trial_data_contract`,
`validate_loaded_dataset`, `verify_loaded_slice`, `verify_trial_tag_lineage`,
`data/run_snapshot.py::load_data_snapshot_for_run` -> `load_dataset_by_trial`.

**Steps:**
- [x] Write failing unit tests for L1-L4 validators.

  Tests must cover:
  - L1 `validate_trial_data_contract(contract, dataset)` raises when dataset id,
    identity hash, target, split id, row count, or column count mismatch.
  - L2 `validate_loaded_dataset(loaded, dataset)` raises for data-content, registry, and schema
    hash drift.
  - L3 `verify_loaded_slice(loaded, slice_contract)` raises for row-count and content-hash drift.
  - L4 `verify_trial_tag_lineage(contract, run_id)` uses only the mlflow seam to read trial tags
    and raises on mismatched `data.dataset_id`, `data.identity_hash`, `data.manifest_uri`, or
    `data.slice.<name>.content_hash`.

  Run:
  ```bash
  uv run pytest tests/unit/data/test_contract_validators.py -v
  ```
  Expected: FAIL because L4 is currently a no-op and some validator edge cases are untested.

- [x] Add minimal MLflow seam reads needed by data replay.

  Implement:
  - `mlflow.trial.get_tags(run_id: str) -> dict[str, str]`
  - `mlflow.trial.artifacts.load_trial_data_contract(run_id: str) -> TrialDataContract`
  - `mlflow.experiment.find_trial_run_id(trial_id: str, *, experiment_id=None) -> str`

  These functions stay inside `automl/mlflow/**`; data code calls the seam and never imports
  PyPI `mlflow`.

- [x] Implement L4 using the new tag read.

  Expected tag keys:
  - `data.dataset_id`
  - `data.identity_hash`
  - `data.manifest_uri`
  - `data.slice.<name>.content_hash`

  A missing tag is an error. This matches the A2 integrity gate and avoids silent replay from
  tampered runs.

- [x] Write failing integration tests for multi-range loads.

  Add a dataset whose `SPLITID` values make rows in two disjoint ranges easy to assert.
  Call:
  ```python
  load_dataset_by_id(loaded.id, split_range=((80, 90), (95, 100)), session=active)
  ```
  Expected result: only rows with buckets in `80..89` or `95..99`, with
  `split_name is None` and `split_ranges == ((80, 90), (95, 100))`.

  Run:
  ```bash
  uv run pytest tests/integration/data_pipeline/test_trial_replay.py::test_load_dataset_by_id_accepts_disjoint_multi_range -v
  ```
  Expected: FAIL if current behavior mishandles multi-range normalization or assertions.

- [x] Implement multi-range loader behavior in `data/registry.py`.

  Keep the existing bare-pair convenience:
  - `(80, 100)` normalizes to `((80, 100),)`.
  - `((80, 90), (95, 100))` remains disjoint.
  - `split_name` and `split_range` remain mutually exclusive.
  - Range validation continues through `project.Splits`.

- [x] Write failing integration tests for `load_dataset_by_trial`.

  Test setup:
  - Materialize the tiny dataset.
  - Create or reuse a small MLflow run through `mlflow.trial.active`.
  - Write a `TrialDataContract` with `splits={"train": ((0, 50),), "holdout": ((90, 100),)}` and
    one `SliceContract` for `train`.
  - Set the L4 tags through the seam.
  - Call `load_dataset_by_trial(trial_id, split_name="train", session=active)`.

  Assertions:
  - It loads by the contract's dataset id.
  - It resolves named splits from the contract, not current `active.config.run_config.splits`.
  - Unknown split names raise `KeyError` naming available contract splits.
  - A mismatched tag makes replay raise `DataError`.

  Run:
  ```bash
  uv run pytest tests/integration/data_pipeline/test_trial_replay.py -v
  ```
  Expected: FAIL until `load_dataset_by_trial` is implemented.

- [x] Implement `load_dataset_by_trial(trial_id, *, split_name=None, split_range=None, session=None)`.

  Behavior:
  - Resolve `trial_id` to run id with `mlflow.experiment.find_trial_run_id`.
  - Load `TrialDataContract` from `mlflow.trial.artifacts.load_trial_data_contract`.
  - Load the contract dataset id.
  - Resolve `split_name` only against `contract.splits`.
  - Run L2 by reusing `load_dataset_by_id`.
  - Run L3 when returning a named contract slice that appears in `contract.slices`.
  - Always run L4 before returning.

- [x] Run contract and replay tests.

  Run:
  ```bash
  uv run pytest tests/unit/data/test_contract_validators.py tests/integration/data_pipeline/test_trial_replay.py -v
  ```
  Expected: PASS.

- [x] Run the local runner integration.

  Run:
  ```bash
  uv run pytest tests/integration/runner/test_one_trial_local.py -v
  ```
  Expected: PASS.

**Acceptance:** A2.3 is covered; final external evidence passed in P2.6.

---

## P2.4 — RequiredTransformer Contract And Home Credit WOE-Gated Trial

**Files:**
- Create: `automl/model/preprocessing.py`
- Modify: `automl/model/base.py`
- Modify: `automl/model/checks.py`
- Modify: `automl/model/__init__.py`
- Modify: `automl/validate/targets.py`
- Modify: `automl/project/config.py`
- Create: `projects/example_homecredit/model/__init__.py`
- Create: `projects/example_homecredit/model/preprocessing.py`
- Modify: `projects/example_homecredit/config.py`
- Modify: `projects/example_homecredit/model.py`
- Test: `tests/unit/model/test_required_transformers.py`
- Test: `tests/unit/validate/test_required_transformer_gate.py`
- Test: `tests/integration/homecredit/test_required_transformer_fixture.py`
- Test: `tests/integration/runner/test_one_trial_local.py`

**Specs:** `spec/06-model.md` Q1-Q7; `spec/01-project-context.md` §3.1;
`spec/04-validate.md` Q1/Q4; `spec/08-runner.md` Q3/Q5.

**Legacy source:**
- `automl_legacy/core/base_model.py`
- `automl_legacy/validate/builtin/model_checks.py`

**Migration rows:** `model/preprocessing.py::RequiredTransformer`,
`SklearnTransformer`, `describe_required_transformers`,
`model/checks.py::check_required_transformers`.

**Steps:**
- [x] Write failing unit tests for `RequiredTransformer` and description output.

  Tests must assert:
  - `RequiredTransformer(name="homecredit_organization_woe", transformer=obj,
    input_cols=["organization_type"])` stores `input_cols` as a tuple/list suitable for cloning.
  - `describe_required_transformers(session)` returns dicts with `name`, `type`,
    `import_path`, and `columns`.
  - Empty or missing project requirements return `[]`.
  - `BaseModel.required_transformer_entries(session=active)` returns
    ColumnTransformer-ready `(name, cloned_transformer, columns)` entries and does not return
    the same transformer instance declared in config.

  Run:
  ```bash
  uv run pytest tests/unit/model/test_required_transformers.py -v
  ```
  Expected: FAIL because `automl/model/preprocessing.py` does not exist.

- [x] Implement `automl/model/preprocessing.py`.

  Include:
  - frozen `RequiredTransformer`
  - runtime-checkable `SklearnTransformer` protocol or simple structural helper for `fit` and
    `transform`
  - `required_transformer_entries(session=None)`
  - `describe_required_transformers(session=None)`

  Use `sklearn.base.clone` for declared transformers.

- [x] Move the inert hook out of `model/base.py`.

  `BaseModel.required_transformer_entries()` delegates to
  `automl.model.preprocessing.required_transformer_entries`. The top-level inert function in
  `base.py` should be removed from exports.

- [x] Tighten `ProjectConfig.required_transformers` type handling.

  `ProjectConfig.load()` continues to treat absent/None as `[]`, but if a non-list is declared it
  raises. If a list contains non-`RequiredTransformer` values, it raises a clear `TypeError`.

- [x] Write failing gate tests for required-transformer validation.

  Test cases:
  - A compliant fitted model whose `self.preprocessor` is a top-level fitted
    `ColumnTransformer` with an entry named `homecredit_organization_woe` over
    `["organization_type"]` passes.
  - A fitted model with no `ColumnTransformer` fails with an Issue whose check is
    `model.required_transformers`.
  - A fitted top-level `Pipeline` wrapping a `ColumnTransformer` fails.
  - A wrong transformer class fails.
  - Missing required columns in the fitted transformer triple fails.
  - Empty requirements produce no issues.

  Run:
  ```bash
  uv run pytest tests/unit/validate/test_required_transformer_gate.py -v
  ```
  Expected: FAIL until `check_required_transformers` is implemented and called.

- [x] Implement `model.checks.check_required_transformers(instance, *, session=None)`.

  Behavior:
  - Resolve ambient session if `session is None`.
  - No-op when `session.config.required_transformers` is empty.
  - Require `instance.preprocessor` to be a top-level `sklearn.compose.ColumnTransformer`.
  - Inspect fitted `transformers_` triples.
  - For each required transformer, check declared name, transformer type, and
    `input_cols ⊆ fitted_columns`.
  - Return canonical `Issue` objects; do not raise for normal gate failures.

- [x] Wire the gate into `validate.model`.

  After successful fit and post-fit attrs check, call:
  ```python
  check_required_transformers(instance=instance)
  ```
  Keep `validate.model(cls, *, df, registry)` unchanged.

- [x] Add the Home Credit project transformer module.

  `projects/example_homecredit/model/preprocessing.py` defines `WOEEncoder` with sklearn-style
  `fit(X, y)` and `transform(X)`. Keep it deterministic and cloudpickle-friendly:
  - class defined at module top level;
  - handles unseen/missing categories with a learned global fallback;
  - returns a 2D numeric array/dataframe;
  - uses `ORGANIZATION_TYPE` after pipeline column normalization, so the declared input column is
    `organization_type`.

- [x] Declare the Home Credit requirement in config.

  In `projects/example_homecredit/config.py`, add:
  ```python
  from automl.model import RequiredTransformer
  from projects.example_homecredit.model.preprocessing import WOEEncoder

  REQUIRED_TRANSFORMERS = [
      RequiredTransformer(
          name="homecredit_organization_woe",
          transformer=WOEEncoder(),
          input_cols=["organization_type"],
      )
  ]
  ```
  Include `required_transformers=REQUIRED_TRANSFORMERS` in `PROJECT_CONFIG`.

- [x] Update the Home Credit model to splice the hook.

  Replace the numeric-only preprocessor with a top-level `ColumnTransformer` whose first entries
  are `*self.required_transformer_entries()` and whose remaining branch handles numeric columns.
  Keep downstream estimator logic in `self.model`, not in a Pipeline wrapping the
  `ColumnTransformer`.

- [x] Add an integration fixture proving both pass and fail paths.

  Tests:
  - `HomeCreditLogisticModel` fits and validates with the declared WOE requirement.
  - A local `OmittingRequiredTransformerModel` that subclasses `BaseModel` but keeps the old
    numeric-only preprocessor fails `validate.model` with `model.required_transformers`.

  Run:
  ```bash
  uv run pytest tests/unit/model/test_required_transformers.py tests/unit/validate/test_required_transformer_gate.py tests/integration/homecredit/test_required_transformer_fixture.py -v
  ```
  Expected: PASS.

- [x] Run the local one-trial path with the WOE requirement declared.

  Run:
  ```bash
  uv run pytest tests/integration/runner/test_one_trial_local.py -v
  ```
  Expected: PASS, and the validation gate no longer treats required transformers as inert.

**Acceptance:** A2.4 is covered; final external evidence passed in P2.6.

---

## P2.5 — Profile And Project-Overview Artifacts

**Files:**
- Create: `automl/data/profile.py`
- Modify: `automl/data/__init__.py`
- Modify: `automl/mlflow/project/overview.py`
- Modify: `automl/mlflow/project/artifacts.py`
- Modify: `automl/mlflow/project/__init__.py`
- Test: `tests/unit/data/test_profile.py`
- Test: `tests/unit/mlflow/test_project_profile_artifacts.py`
- Test: `tests/integration/data_pipeline/test_profile_integration.py`

**Specs:** `spec/05-data.md` Q5, §5 profile; `spec/02-mlflow-seam.md` §6.1, §9;
`spec/00-structural-design.md` §11.1 data profile verb.

**Legacy source:**
- `automl_legacy/profile/core.py`
- `automl_legacy/profile/snapshot.py`
- `automl_legacy/mlflow/store.py` overview helpers

**Migration rows:** `automl/profile/*` -> `data/profile.py`,
`ProfileResult` -> `Profile`, MLflow profile publishing -> `mlflow/project/artifacts.py`.

**Steps:**
- [x] Write failing profile unit tests.

  Tests must assert:
  - `Profile` has `dataset_id`, `target_column`, `data_card_uri`,
    `data_observations_uri`, `profile_manifest_uri`, `chart_uris`, `created_at`,
    `schema_version`.
  - `Profile.to_dict()` / `from_dict()` round-trip and strip unknown keys.
  - Pure profile helpers produce `data_card.json`, `data_observations.json`,
    `profile_manifest.json`, and chart PNGs for a tiny loaded dataset.
  - Per-check/chart exception wrapping means one crashing chart function records an observation
    issue and does not abort the whole profile.

  Run:
  ```bash
  uv run pytest tests/unit/data/test_profile.py -v
  ```
  Expected: FAIL because `automl/data/profile.py` does not exist.

- [x] Implement `data/profile.py`.

  Keep it single-file:
  - private `_STATS_CHECKS`
  - private `_CHARTS`
  - pure deterministic stats/chart helpers ported from legacy profile core as needed
  - public `Profile`
  - public `profile(dataset_id=None, *, session=None) -> Profile`
  - public `get_profile(dataset_id=None, *, session=None) -> Profile | None`

  `profile()` loads the active dataset or requested dataset with `load_dataset_by_id`, writes a
  temporary local artifact directory, and delegates durable writes to `mlflow.project.artifacts`.

- [x] Write failing MLflow seam tests for project overview profile artifacts.

  Tests must assert:
  - `mlflow.project.ensure_overview()` creates or returns a project overview run.
  - `mlflow.project.artifacts.write_profile(dataset_id, local_dir=...)` logs the profile files
    under `<dataset_id>/profile/` on the project-overview run and returns URI strings.
  - `read_profile(dataset_id)` returns a typed `Profile` or `None`.

  Run:
  ```bash
  uv run pytest tests/unit/mlflow/test_project_profile_artifacts.py -v
  ```
  Expected: FAIL because overview/profile writers are Phase 1 placeholders.

- [x] Implement the minimal project-overview seam needed for profile.

  Add only what A2.5 requires:
  - `ensure_overview() -> ProjectOverview` creates `<project>/overview` and an `overview` run.
  - `read_overview()` returns `ProjectOverview | None`.
  - `write_profile(dataset_id, *, local_dir)` logs local files under `<dataset_id>/profile/`.
  - `read_profile(dataset_id)` reads `profile_manifest.json` from the overview run.

  Do not implement experiment views, leaderboard, compare, or broad project metadata in this task.

- [x] Write a profile integration test over a materialized fake-GCS dataset.

  Test flow:
  - Materialize a tiny dataset.
  - Call `profile(session=active)`.
  - Assert returned URIs include `data_card`, `data_observations`, `profile_manifest`, and at
    least one chart.
  - Assert `get_profile(loaded.id, session=active)` returns the same dataset id.

  Run:
  ```bash
  uv run pytest tests/integration/data_pipeline/test_profile_integration.py -v
  ```
  Expected: PASS after implementation.

- [x] Run the local runner integration.

  Run:
  ```bash
  uv run pytest tests/integration/runner/test_one_trial_local.py -v
  ```
  Expected: PASS.

**Acceptance:** A2.5 is covered; final external evidence passed in P2.6.

---

## P2.6 — Phase 2 External Gate

**Files:**
- Create/modify: `tests/e2e/test_phase2_data_model_breadth.py`
- Update after evidence: `docs/superpowers/automl-refactor/plan/acceptance-checklist.md`

**Specs:** `plan/acceptance-checklist.md` A2.1-A2.5.

**Steps:**
- [x] Write the external-gated Phase 2 e2e test.

  Gate behavior:
  - Skip unless `AUTOML_PHASE2_E2E`, `GCS_BUCKET`, `GCP_PROJECT`, and
    `MLFLOW_TRACKING_URI` are present.
  - `use_project("example_homecredit")`.
  - `materialize()`.
  - Assert A2.1 source/index basics in the real project.
  - Assert A2.2 registry has no `golden`/`weak`, supports derived lineage in a trial-local copy,
    and strict constant-drop is covered by unit tests.
  - Assert A2.3 multi-range `load_dataset_by_id` returns a union slice.
  - `run_trial("example_homecredit")` with declared WOE requirement.
  - Assert the run finishes and has real AUC, data contract, eval, model artifacts, and
    required-transformer validation did not fail.
  - Call `profile(session=active)` and assert project-overview profile artifacts exist.
  - Call `load_dataset_by_trial(result.trial_id, split_name=train_split, session=active)` and
    assert L3/L4 pass.

- [x] Run all local tests first.

  Run:
  ```bash
  uv run pytest tests/unit tests/contracts tests/integration -v
  ```
  Expected: PASS.
  Evidence (2026-05-27): `159 passed, 2 warnings`.

- [x] Run the Phase 2 external gate against the original MLflow server.

  Run:
  ```bash
  set -a
  source .env
  set +a
  export MLFLOW_TRACKING_URI=http://127.0.0.1:54321
  export AUTOML_PHASE2_E2E=1
  uv run pytest tests/e2e/test_phase2_data_model_breadth.py -v
  ```
  Expected: PASS. Do not use the temporary Phase 1 verification server as the default.
  Evidence (2026-05-27): `1 passed, 11 warnings`.

- [x] Run architecture ratchets again.

  Run:
  ```bash
  uv run pytest tests/contracts -v
  rg 'automl_legacy' automl projects tests
  rg '(^|\s)(import|from) mlflow' automl projects tests
  ```
  Expected: contract tests pass; no new forbidden imports.
  Evidence (2026-05-27): `uv run pytest tests/contracts -v` -> `9 passed`; import scans found
  no new `automl_legacy` imports and no PyPI `mlflow` imports outside `automl/mlflow/**`.

- [x] Only after the external gate passes, mark A2.1-A2.5 `[x]` with command evidence.

**Acceptance:** A2.1-A2.5 have fresh command evidence.

---

## P2.7 — Phase 2 Docs Closeout

**Files:**
- Modify: `docs/superpowers/automl-refactor/README.md`
- Modify: `docs/superpowers/automl-refactor/plan/README.md`
- Modify: `docs/superpowers/automl-refactor/plan/implementation-strategy.md` only if boundaries changed.
- Modify: `docs/superpowers/automl-refactor/plan/acceptance-checklist.md`
- Modify: `docs/superpowers/automl-refactor/plan/migration-checklist.md`
- Modify: `docs/superpowers/automl-refactor/plan/phases/phase-2-data-model-breadth.md`

**Steps:**
- [x] Update the front-door README status: Phase 2 done, next action is Phase 3 planning.
- [x] Update `plan/README.md` with Phase 2 command evidence and next action.
- [x] Update `acceptance-checklist.md` A2.1-A2.5 only after evidence exists.
- [x] Flip covered migration rows from `[ ]` to `[x]` or leave them open with explicit notes if a
  sub-symbol remains out of Phase 2.
- [x] Update this phase plan's checkboxes as each task lands.
- [x] Run a stale-status scan.

  Run:
  ```bash
  rg -n "Phase 2|A2\.|NEXT ACTION|54322|temporary Phase 1" docs/superpowers/automl-refactor
  ```
  Expected: no stale "Phase 2 next" language after closeout; no future-gate defaults pointing at
  `54322`.
  Evidence (2026-05-27): scan output contains only Phase 2 completion/history references,
  generic `NEXT ACTION` documentation references, and the historical Phase 1 `54322` evidence
  explicitly marked non-default.

**Acceptance:** docs and checklists match implementation evidence before handoff.

---

## Explicitly Deferred Out Of Phase 2

- Eval breadth: `LogLoss`, `ThresholdSweep`, external eval, augmentation, predictions,
  `EvalIndex` durability.
- Experiment and trial views: leaderboard, compare, summary, `trial show`, model load.
- Cleanup cascade and `--hard-delete`.
- Agent loop, hooks, proposer/coder required-preprocessing handoff, and proposal schema changes.
- CLI catalog breadth beyond any tiny profile/data command hook needed by A2.5.
- Full Snowflake execution. Phase 2 includes a resolvable `SnowflakeSource` stub only.
- Broad pyarrow predicate pushdown and physical parquet partitioning.
- Source-trace project-overview logging unless the reviewed Phase 2 gate is expanded to require
  it.

## Self-Review Checklist

- [x] Each A2 acceptance row maps to at least one task:
  A2.1 -> P2.1; A2.2 -> P2.2; A2.3 -> P2.3; A2.4 -> P2.4/P2.6; A2.5 -> P2.5/P2.6.
- [x] Every task has a failing-test step before implementation steps.
- [x] No task introduces an `automl_legacy` import.
- [x] No task imports PyPI `mlflow` outside `automl/mlflow/**`.
- [x] Phase 1 one-trial integration runs after each production task.
- [x] External gate uses `http://127.0.0.1:54321`, not the temporary Phase 1 server.
