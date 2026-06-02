# Phase 1 — Walking Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development`
> (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** prove the new four-layer architecture by running one real Home Credit trial through
the fresh `automl/` package: project context -> MLflow seam -> data -> model/eval/validate ->
runner.

**Architecture:** contract-first vertical slice. Phase 1 starts by pinning the harness and
architectural ratchets, then ports the smallest code needed for the gate. It intentionally avoids
breadth: no agent loop, no experiment views, no cleanup cascade, no profile, no WOE/required
transformer gate, no old-state compatibility, and no broad CLI.

**Tech Stack:** Python 3.11 via `uv`; pytest; pandas/numpy/pyarrow; scikit-learn
`LogisticRegression`; cloudpickle; original local MLflow at `http://127.0.0.1:54321` with the
local `.env` loaded in this refactor worktree, copied from
`/Users/zhengisamazing/1.python_dir/brigit/automl_dev/.env`; GCS bucket
`gs://automl-homecredit-kaggle-wliu`.

**Reviewed/re-cut:** 2026-05-27. The previous T0->T11 module-port plan was rejected after
fresh review because it put contract tests and the real harness too late, overbuilt several
domains before proving the end-to-end path, and contradicted runner spec 08's fit-slice-only
contract.

**Acceptance:** `plan/acceptance-checklist.md` rows **A1.1-A1.4**.

---

## Non-Negotiable Decisions

- New code must never import `automl_legacy`.
- Only `automl/mlflow/**` may import the PyPI `mlflow` package.
- `TrialRef.trial_id` and `TrialRef.run_id` are distinct: `trial_id` is `<number>_<slug>`;
  `run_id` is the MLflow UUID.
- Runner loads only the fit slice for training. Eval data is prepared as a recipe and loaded by
  `evaluate()`.
- Phase 1 model must survive real Home Credit columns: select numeric columns, impute missing
  values, fit LogisticRegression, expose `predict_proba`.
- Phase 1 implements only single-range fit/eval splits; full multi-range loader later landed in
  Phase 2.
- Legacy tests are behavior references, not migration inventory. Rebuild fresh tests that protect
  the new contracts.

---

## Task DAG

```
P1.0 env/preflight
  -> P1.1 contract ratchets
  -> P1.2 Home Credit model fixture
  -> P1.3 leaf primitives
  -> P1.4 project context + facade
  -> P1.5 MLflow seam
  -> P1.6 data thin path
  -> P1.7 model/eval/validate thin path
  -> P1.8 runner + A1 gate
```

Each task is a fresh subagent boundary. The controller reviews spec compliance and code quality
before moving to the next task. If any task surfaces a real spec conflict, stop implementation,
update the relevant spec/plan/checklist with the proposed correction, and get user sign-off if
the correction changes behavior.

---

## P1.0 — Environment And Preflight

**Files:**
- No production files.
- May create: `tests/contracts/test_environment.py`

**Specs:** README working norms; `plan/implementation-strategy.md` §1, §5.

**Legacy source:** none.

**Steps:**
- [x] Run `uv sync` in `/Users/zhengisamazing/1.python_dir/brigit/automl_dev-refactor`.
- [x] Run `uv run python -c "import automl; print(automl.__version__)"`.
- [x] Add a contract test that imports required Phase 1 runtime dependencies:
  `pandas`, `numpy`, `pyarrow`, `sklearn`, `cloudpickle`, `mlflow`, `google.cloud.storage`.
- [x] Run `uv run pytest tests/contracts/test_environment.py -v`.
- [x] Record any external-service gaps as preflight notes, not as passing A1 rows.

**Acceptance:** env works and dependency availability is pinned by a contract test.

---

## P1.1 — Architectural Contract Ratchets

**Files:**
- Create: `tests/contracts/test_architecture.py`
- Create: `tests/contracts/test_pytest_structure.py`

**Specs:** `spec/00-structural-design.md` §7, §8, §9, §10, §13.7.

**Legacy source:** none; this pins new architecture, not legacy behavior.

**Steps:**
- [x] Write contract tests for:
  - fresh package folder shape;
  - no imports from `automl_legacy` in `automl/`, `projects/`, or new `tests/`;
  - no direct `import mlflow` / `from mlflow` outside `automl/mlflow/`;
  - domain import boundaries for the Phase 1 slice;
  - pytest default `testpaths` includes `tests/unit` and `tests/contracts`.
- [x] Run `uv run pytest tests/contracts -v` and confirm failures identify missing
  implementation, not broken tests.
- [x] Make only the minimal non-production or import-boundary fixes needed for the contracts that
  should already hold.
- [x] Keep these tests green after every later task.

**Acceptance:** A1.4 ratchets exist at the beginning of Phase 1.

---

## P1.2 — Home Credit Model Fixture

**Files:**
- Create: `projects/example_homecredit/model.py`
- Create: `tests/integration/homecredit/test_model_fixture.py`

**Specs:** `spec/06-model.md`, `spec/08-runner.md`.

**Legacy source:**
- `automl_legacy/core/base_model.py`

**Steps:**
- [x] Write the failing fixture test first. It should import the project model class and prove it
  can fit/predict on a tiny DataFrame with numeric columns plus missing values.
- [x] Add `model.py` with a minimal Home Credit model:
  numeric feature selection, `SimpleImputer`, `LogisticRegression(max_iter=...)`, `fit`,
  `predict`, and `predict_proba`.
- [x] Run `uv run pytest tests/integration/homecredit/test_model_fixture.py -v`.

**Acceptance:** the model half of the harness is explicit before library ports. The typed
`config.py` wiring lands in P1.7 after the contracts it imports exist, but still before runner.

---

## P1.3 — Leaf Primitives

**Files:**
- Existing: `automl/errors.py`
- Create: `automl/utils/hashing.py`
- Create: `automl/utils/io/gcs.py`
- Create: `automl/utils/paths.py`
- Create: `automl/utils/logging.py`
- Create/modify exports: `automl/utils/__init__.py`, `automl/utils/io/__init__.py`
- Tests under: `tests/unit/utils/`, `tests/unit/test_errors.py`

**Specs:** `spec/00-structural-design.md` §10, §13.8; `spec/05-data.md` §5 hash primitives.

**Legacy source:**
- `automl_legacy/data/snapshot.py`
- `automl_legacy/io/gcs.py`
- `automl_legacy/utils/logging.py`
- `automl_legacy/mlflow/artifacts/gcs_paths.py` for URI conventions only.

**Steps:**
- [x] Test and pin the existing errors hierarchy.
- [x] Test `json_hash`, `schema_hash`, and `dataframe_content_hash` for deterministic,
  dtype/order-sensitive behavior. Port from legacy snapshot hashing rather than the earlier
  simplified pseudocode.
- [x] Test GCS URI parsing/joining and JSON/parquet read/write through monkeypatched storage
  clients or local fakes. Do not require real GCS in unit tests.
- [x] Add path/logging helpers only where required by Phase 1.
- [x] Run `uv run pytest tests/unit/test_errors.py tests/unit/utils -v`.

**Acceptance:** leaf code is small, deterministic, and independent of domains.

---

## P1.4 — Project Context, Run Config, And Early Facade

**Files:**
- Create: `automl/project/config.py`
- Create: `automl/project/session.py`
- Create: `automl/project/run_config.py`
- Create: `automl/project/_import.py`
- Modify: `automl/project/__init__.py`
- Modify: `automl/__init__.py`
- Tests under: `tests/unit/project/`

**Specs:** `spec/01-project-context.md`; `spec/00-structural-design.md` §12;
`spec/05-data.md` Q8 for `Splits` / `RunConfig` split semantics.

**Legacy source:**
- `automl_legacy/core/project_context.py`
- `automl_legacy/core/run_config.py`
- `automl_legacy/core/task.py`
- `automl_legacy/core/config.py`

**Steps:**
- [x] Write tests for `Splits`, `RunConfig`, `ProjectConfig`, `Session`, `use_project`,
  `session`, `active_session`, `clear_session`, and `update_session`.
- [x] Implement the minimal config/session objects with `dry_run=False`, `namespace=""`, and
  `experiment_id` threading.
- [x] Implement `_bind_mlflow_for(session)` as a late import so project does not create an
  import cycle.
- [x] Export the early Tier-1 facade from `automl/__init__.py`.
- [x] Run `uv run pytest tests/unit/project -v` plus contracts.

**Acceptance:** project/session plumbing exists before MLflow/data/runner rely on it.

---

## P1.5 — MLflow Seam Thin Path

**Files:**
- Create: `automl/mlflow/client.py`
- Create: `automl/mlflow/_routing.py`
- Create: `automl/mlflow/tags.py`
- Create: `automl/mlflow/experiment/lifecycle.py`
- Create: `automl/mlflow/experiment/queries.py`
- Create: `automl/mlflow/project/artifacts.py`
- Create: `automl/mlflow/project/overview.py`
- Create: `automl/mlflow/trial/lifecycle.py`
- Create: `automl/mlflow/trial/logging.py`
- Create: `automl/mlflow/trial/artifacts/data.py`
- Create: `automl/mlflow/trial/artifacts/eval.py`
- Create: `automl/mlflow/trial/artifacts/model.py`
- Create: `automl/mlflow/trial/artifacts/manifest.py`
- Modify: `automl/mlflow/trial/artifacts/__init__.py`
- Modify: `automl/mlflow/**/__init__.py`
- Tests under: `tests/unit/mlflow/`

**Specs:** `spec/02-mlflow-seam.md`; runner call shape from `spec/08-runner.md`.

**Legacy source:**
- `automl_legacy/mlflow/store.py`
- `automl_legacy/trial/creation.py` (`_next_trial_number_from_mlflow`)
- `automl_legacy/mlflow/artifacts/gcs_paths.py`
- `automl_legacy/mlflow/artifacts/*`

**Steps:**
- [x] Test `bind()/bound()` context behavior and route strings:
  real, dry_run, namespace, namespace+dry_run.
- [x] Test file-backed MLflow `ensure_experiment`, `next_trial_number`, `active()` context,
  metric logging, JSON artifact logging, and model/artifact logging.
- [x] Implement only seam functions needed by Phase 1.
- [x] Keep all PyPI `mlflow` imports inside `automl/mlflow/`.
- [x] Run `uv run pytest tests/unit/mlflow tests/contracts -v`.

**Acceptance:** A1.2 seam plumbing has a unit-level proof before runner integration.

---

## P1.6 — Data Thin Path

**Files:**
- Create: `automl/data/spec.py`
- Create: `automl/data/dataset.py`
- Create: `automl/data/features.py`
- Create: `automl/data/sources/base.py`
- Create: `automl/data/sources/local_csv.py`
- Create: `automl/data/pipeline.py`
- Create: `automl/data/registry.py`
- Create: `automl/data/contract.py`
- Modify: `automl/data/__init__.py`, `automl/data/sources/__init__.py`
- Tests under: `tests/unit/data/`, `tests/integration/data_pipeline/`

**Specs:** `spec/05-data.md`; `spec/08-runner.md` Q4/Q5 for fit-slice-only contract.

**Legacy source:**
- `automl_legacy/data/spec.py`
- `automl_legacy/data/sources.py`
- `automl_legacy/data/pipeline.py`
- `automl_legacy/data/snapshot.py`
- `automl_legacy/data/snapshots.py`
- `automl_legacy/data/split.py`
- `automl_legacy/data/run_snapshot.py`
- `automl_legacy/core/feature_registry.py`

**Steps:**
- [x] Test `LocalCSVSource` + `DataPipeline` on the committed Home Credit sample.
- [x] Test `materialize()` creates a `Dataset` manifest, `LoadedDataset`, feature registry, split
  id, hashes, and an index entry.
- [x] Test `materialize()` writes `feature_registry.csv`, keeps `dataset_index.json`
  project-scoped (no persisted `active_dataset_id`), flips the experiment active-dataset seam,
  is idempotent for complete existing Dataset objects, and raises `StorageError` for partial
  Dataset objects.
- [x] Test `load_dataset(split_name="train")` and `load_dataset_by_id(..., split_name="test")`
  return `LoadedSlice`.
- [x] Test L2 runs by default and a corrupted manifest/hash raises `DataError`.
- [x] Mark `load_dataset_by_trial()` plus L3/L4 validation as deferred for A1; those later
  landed in Phase 2. Source trace artifact logging and broad pyarrow predicate pushdown remain
  deferred. Idempotence and
  partial-object protection are not deferred.
- [x] Document the Phase 1 active-dataset seam simplification: the runtime active Dataset is
  stored as `automl.active_dataset_id` on the routed MLflow experiment via
  `mlflow.experiment.get_active_dataset` / `set_active_dataset`, not yet on a separate
  experiment-overview run. This preserves the P1.6 contract that `dataset_index.json` stays
  project-scoped and active population is runtime/seam-owned; the full overview-run home remains
  a later seam-breadth cleanup.
- [x] Test `TrialDataContract`/`TrialRef` includes both `trial_id` and `run_id`.
- [x] Run `uv run pytest tests/unit/data tests/integration/data_pipeline -v`.

**Acceptance:** A1.3 has a local proof; data APIs are enough for runner and eval. Deferred beyond
A1: `load_dataset_by_trial()`/L3/L4 later landed in Phase 2; source trace logging and broad
pyarrow pushdown remain deferred.
Required in P1.6: CSV registry persistence, project-scoped index persistence, active-dataset seam
population, complete-object idempotence, and partial-object refusal. Phase 1 uses an MLflow
experiment tag for the active-dataset seam; moving that pointer behind a real experiment-overview
run is not required for A1.

---

## P1.7 — Model, Eval, And Validate Thin Path

**Files:**
- Create: `automl/model/base.py`
- Create: `automl/model/packaging.py`
- Create: `automl/model/checks.py`
- Modify: `automl/model/__init__.py`
- Create: `automl/eval/base.py`
- Create: `automl/eval/metrics.py`
- Create: `automl/eval/results.py`
- Create: `automl/eval/eval_dataset.py`
- Create: `automl/eval/prepare.py`
- Create: `automl/eval/evaluate.py`
- Create: `automl/eval/_load.py`
- Modify: `automl/eval/__init__.py`
- Create: `automl/validate/base.py`
- Create: `automl/validate/targets.py`
- Create: `automl/validate/synthetic.py`
- Modify: `automl/validate/__init__.py`
- Modify: `projects/example_homecredit/config.py`
- Create: `tests/integration/homecredit/test_harness_fixture.py`
- Tests under: `tests/unit/model/`, `tests/unit/eval/`, `tests/unit/validate/`

**Specs:** `spec/04-validate.md`, `spec/06-model.md`, `spec/07-eval.md`,
`spec/08-runner.md` Q5.

**Legacy source:**
- `automl_legacy/core/base_model.py`
- `automl_legacy/runner/_execute.py` for model saving behavior
- `automl_legacy/eval/{base,metrics,evaluate,snapshot,loader,loading,runner}.py`
- `automl_legacy/validate/{types,targets,synthetic,builtin/model_checks}.py`
- `projects/example_homecredit/config.py`
- `automl_legacy/core/run_config.py`

**Steps:**
- [x] Test minimal `BaseModel` behavior and `save_model` cloudpickle round-trip.
- [x] Implement inert `required_transformer_entries(session=None) -> []`; full transformer gate
  later landed in Phase 2.
- [x] Test `Auc`, `EvalSpec`, `prepare_eval_dataset`, and `evaluate()` using a split-view recipe
  backed by `data.load_dataset_by_id`.
- [x] Keep the P1.7 eval dataset path intentionally thin: split-view identities are
  process-local recipes keyed by `(dataset_id, split name)` for the A1 runner handoff. Durable
  eval manifests and concrete bucket-range recipe identities remain later eval breadth.
- [x] Test `validate.model` catches fit/predict failures and returns `ValidationReport`.
- [x] Keep `validate.project` and `validate.proposal` out of the A1 happy path; they must not be
  silent no-ops while their real checks are deferred.
- [x] Update the Home Credit fixture config to use the new contracts (`ProjectConfig`,
  `DataSpec`, `LocalCSVSource`, `RunConfig`, `Splits`, `EvalSpec`, `Auc`) and point at the
  committed `projects/example_homecredit/data/application_train_sample.csv` by default.
- [x] Add/run the fixture import test asserting `PROJECT_CONFIG`, `train_split="train"`, and
  `eval_split="test"`.
- [x] Run `uv run pytest tests/unit/model tests/unit/eval tests/unit/validate -v`.

**Acceptance:** runner can validate, fit, save, prepare eval, and score real AUC.

---

## P1.8 — Runner And A1 Gate

**Files:**
- Create: `automl/runner/trial.py`
- Create: `automl/runner/paths.py`
- Create: `automl/runner/contract.py`
- Create: `automl/runner/session_lock.py`
- Modify: `automl/runner/__init__.py`
- Create: `tests/integration/runner/test_one_trial_local.py`
- Create/enable: `tests/e2e/test_phase1_walking_skeleton.py`
- Update after evidence: `docs/superpowers/automl-refactor/plan/acceptance-checklist.md`
- Update after evidence: `docs/superpowers/automl-refactor/README.md`

**Specs:** `spec/08-runner.md`, plus contracts from specs 01/02/05/06/07.

**Legacy source:**
- `automl_legacy/runner/_execute.py`
- `automl_legacy/runner/_stages.py`
- `automl_legacy/trial/creation.py`

**Steps:**
- [x] Write the integration test first against a test-owned project copy and file-backed MLflow.
- [x] Implement `run_trial(path_or_project, *, session=None)` as a straight-line Phase 1 chain:
  load fit slice -> pre-fit validate -> ensure experiment/next trial id -> open MLflow run ->
  fit model -> save model -> prepare eval dataset -> call `evaluate()` -> build/log
  `TrialDataContract` -> log eval/model artifacts -> return `TrialResult`.
- [x] Keep runner contract fit-slice-only. Do not eagerly load eval data in runner.
- [x] Run `uv run pytest tests/integration/runner/test_one_trial_local.py -v`.
- [x] Run the external Home Credit A1 gate with local MLflow + GCS configured.
- [x] Only after the A1 gate passes, flip A1.1-A1.4 to `[x]` and update the front-door status for
  the then-current next phase.

**Acceptance:** A1.1-A1.4 have fresh command evidence. No acceptance row flips before evidence.

---

## Phase 1 Review Checklist

- [x] `uv run pytest tests/unit tests/contracts -v`
- [x] `uv run pytest tests/integration/data_pipeline tests/integration/homecredit tests/integration/runner -v`
- [x] External-gated Home Credit e2e command run with local MLflow + GCS configured.
- [x] `rg 'automl_legacy' automl projects tests` shows no new-code imports; current matches are
  the architecture ratchet and the explanatory docstring in `automl/__init__.py`.
- [x] `rg '(^|\\s)(import|from) mlflow' automl projects tests` shows PyPI `mlflow` only under
  `automl/mlflow/` and tests that intentionally verify the seam.
- [x] A1.1-A1.4 checklist rows are updated only after the gate passes.

---

## Deferred Explicitly To Later Phases

- Multi-range loader breadth beyond the A1 split needs — resolved in Phase 2.
- Snowflake and GCS parquet sources — resolved in Phase 2, with Snowflake execution still stubbed.
- Profile — resolved in Phase 2.
- Required transformer gate / WOE example — resolved in Phase 2.
- LogLoss, threshold sweep, external eval, prediction index (Phase 3).
- Trial/experiment read models, cleanup cascade, agent loop, and full CLI (Phases 4-6).
