# Acceptance Checklist — AutoML Refactor (behavior-level)

**Purpose:** the **behavior-level** complement to `migration-checklist.md` (which tracks
*symbols*). This tracks *capabilities* — "can the new `automl/` package actually do X against the
Home Credit harness?" — phase by phase. The migration checklist proves *nothing was left
behind*; this proves *the thing still works*.

**Authority:** `implementation-strategy.md` defines the phases + gates; this file is the
runnable gate per phase. A phase is **done** only when its row(s) here are green *and* its
migration-checklist rows are `[x]`/`[-]`.

**Harness:** Home Credit, run in an `automl_runs/<dataset>-<seq>/` working copy (never the
`../kaggle_home_credit/` base sandbox). Local MLflow convention is
`http://127.0.0.1:54321` (`mlflow_local start`). When running from this refactor worktree, first
load the local `.env`, copied from `/Users/zhengisamazing/1.python_dir/brigit/automl_dev/.env`;
git worktrees may not carry the auth settings needed by the original MLflow server. GCS
`gs://automl-homecredit-kaggle-wliu`; `local_csv` data adapter via the data-pipeline override
hook. See the sibling workspace
`/Users/zhengisamazing/1.python_dir/brigit/kaggle_home_credit/README.md` for setup. Phase 1 also
commits a small new-style project fixture under this worktree so the gate does not depend on old
generated project layout.

**Status legend:** `[ ]` not started · `[~]` in progress · `[x]` passing · `[!]` blocked.

---

## Phase 0 — pre-flight

- [x] **A0.1** Completeness audit done; migration-checklist coverage verified (one gap filled). *(2026-05-27)*
- [x] **A0.2** This `acceptance-checklist.md` authored.
- [x] **A0.3** Scaffold done *(2026-05-27, commit `281ef80` on `refactor/four-layer`)*: worktree `../automl_dev-refactor` off clean HEAD; `git mv automl → automl_legacy` + `tests → tests_legacy` (frozen reference; topology correction — the repo root *is* `automl_dev/`, so the rename applies to the package, not the repo dir); fresh `automl/` four-layer skeleton + `tests/` layout; `pyproject` excludes `automl_legacy*`; refactor docs moved in-repo to `docs/superpowers/automl-refactor/`. *(Deferred to Phase 1: the `tests/contracts/test_pytest_structure.py` ratchet + `testpaths` expansion — authored fresh when the first tests land, since the legacy ratchet is frozen in `tests_legacy/`.)*
- [x] **A0.4** `import automl` succeeds on the fresh skeleton (root + nested subpackages + `errors`). *(Verified with `python3`; the full `uv sync` project env is built at the start of Phase 1, when real deps are first needed — no deps required to import the empty skeleton.)*

## Phase 1 — walking skeleton (the vertical slice) ★

- [x] **A1.1 — one real trial runs end-to-end.** A real-but-simple model (numeric column selection + imputation + LogisticRegression with `predict_proba`) fits on real Home Credit data via `runner.run_trial`: loads data (`local_csv` source → `materialize`/`load_dataset`), pre-fit validates, opens an MLflow run, prepares the eval dataset, computes real **AUC** through `evaluate()`, logs metric + writes the **TrialDataContract + EvalResult + model** artifacts to local MLflow + GCS. Exit 0; run visible in MLflow; artifacts present in GCS.
  - P1.7/P1.8 A1 uses a split-view eval dataset recipe held process-locally by
    `(dataset_id, split name)`. Durable eval dataset manifests and concrete bucket-range recipe
    identity are later eval breadth, not required for the first walking skeleton.
  - Evidence (2026-05-27): `AUTOML_PHASE1_E2E=1 MLFLOW_TRACKING_URI=http://127.0.0.1:54322 GCS_BUCKET=automl-homecredit-kaggle-wliu GCP_PROJECT=fluent-imprint-458323-k4 GCS_PREFIX=phase1-e2e-<timestamp> uv run pytest tests/e2e/test_phase1_walking_skeleton.py -v` -> `1 passed`. This is historical evidence only; future gates should use the original `54321` MLflow server with the original `.env` loaded.
- [x] **A1.2 — `session` + seam plumbing proven.** `use_project` binds the mlflow seam; the trial routes to the correct `<route>` (real universe); `next_trial_number` assigns from the seam.
- [x] **A1.3 — L2 integrity holds.** `load_dataset_by_id` runs L2 (loaded↔manifest) by default; a deliberately corrupted manifest makes the load raise. Dataset materialization persists `feature_registry.csv`, stores only project-scoped `schema_version` + `datasets` in `dataset_index.json`, populates the runtime active Dataset from the experiment seam, reuses complete existing Dataset objects without rewriting them, and refuses partial Dataset objects with `StorageError`. Phase 1 stores the active-Dataset pointer as an MLflow experiment tag. Phase 2 later added `load_dataset_by_trial()`/L3/L4 and multi-range loader breadth; follow-up fixes added source trace hook consumption and namespace/dry-run Dataset routing. Broad pyarrow pushdown remains deferred.
- [x] **A1.4 — contract tests green.** Architectural invariants pinned early and kept green: domain import boundaries, `session` convention, seam-only mlflow access, no `automl_legacy` imports, four-layer shape, pytest path policy.

## Phase 2 — data & model breadth

- [x] **A2.1 — data source/index breadth.** `local_csv` + Snowflake-stub + gcs_parquet resolve (harness uses `local_csv`); `build_dataset`/`list_datasets` work; `DatasetIndex` registers; the Phase 1 one-trial path stays green.
  - Evidence (2026-05-27): `uv run pytest tests/unit/data/test_sources_breadth.py tests/integration/data_pipeline/test_materialize_load.py -v` -> `9 passed`; `uv run pytest tests/integration/runner/test_one_trial_local.py -v` -> `2 passed`.
- [x] **A2.2 — FeatureRegistry breadth.** `derived` / `source_columns` / `add_derived` work; `constant_drop_threshold=1.0` actually drops a strict-constant column.
  - Evidence (2026-05-27): `uv run pytest tests/unit/data/test_feature_registry_breadth.py tests/unit/data/test_sources_pipeline_contract.py -v` -> `15 passed`.
- [x] **A2.3 — full L1–L4 validators + multi-range loader.** `load_dataset_by_id(split_range=((80,90),(95,100)))` returns the union slice; `load_dataset_by_trial` runs L3+L4.
  - Evidence (2026-05-27): `uv run pytest tests/unit/data/test_contract_validators.py tests/integration/data_pipeline/test_trial_replay.py -v` -> `14 passed`.
- [x] **A2.4 — project-mandated transformer gate.** A trial declaring `REQUIRED_TRANSFORMERS` (real Home Credit `WOEEncoder` on `ORGANIZATION_TYPE`) passes the `check_required_transformers` gate; a model omitting it **fails** validation with a clear Issue.
  - Evidence (2026-05-27): `uv run pytest tests/unit/model/test_required_transformers.py tests/unit/validate/test_required_transformer_gate.py tests/integration/homecredit/test_required_transformer_fixture.py -v` -> `13 passed`; `uv run pytest tests/integration/homecredit -v` -> `7 passed`.
- [x] **A2.5 — profile runs.** `data.profile` produces the project observations + EDA charts, written to the project-overview run.
  - Evidence (2026-05-27): `uv run pytest tests/unit/data/test_profile.py tests/unit/mlflow/test_project_profile_artifacts.py tests/integration/data_pipeline/test_profile_integration.py -v` -> `7 passed`.

Phase 2 gate evidence (2026-05-27): `uv run pytest tests/unit tests/contracts tests/integration -v` -> `159 passed`; `AUTOML_PHASE2_E2E=1 MLFLOW_TRACKING_URI=http://127.0.0.1:54321 uv run pytest tests/e2e/test_phase2_data_model_breadth.py -v` -> `1 passed`; `uv run pytest tests/contracts -v` -> `9 passed`. Import ratchets found no new `automl_legacy` imports and no PyPI `mlflow` imports outside `automl/mlflow/**`.

## Phase 3 — eval breadth

- [x] **A3.1 — all metrics compute.** `Auc` / `LogLoss` / `ThresholdSweep` produce values; the locked set logs under namespaced `<label>.<metric>` keys.
- [x] **A3.2 — external eval + augmentation.** A trial evaluated against an `external` EvalDataset and an `Augmentation`; `Predictions` + `EvalIndex` persisted; multi-instance eval (`label`) round-trips.

Phase 3 gate evidence (2026-05-28): `uv run pytest tests/unit tests/contracts tests/integration -v` -> `191 passed`; `AUTOML_PHASE3_E2E=1 MLFLOW_TRACKING_URI=http://127.0.0.1:54321 uv run pytest tests/e2e/test_phase3_eval_breadth.py -v` -> `1 passed`; `uv run pytest tests/contracts -v` -> `9 passed`. Import ratchets found no new `automl_legacy` imports and no PyPI `mlflow` imports outside `automl/mlflow/**`.

## Phase 4 — experiment & trial domains

- [x] **A4.1 — leaderboard + compare.** Over several trials, `experiment leaderboard` ranks by `config.primary_metric` and reports `n_unscored`; `experiment compare <id1> <id2>` returns `ComparisonResult` with `MetricDelta`s.
- [x] **A4.2 — trial reads.** `trial show <run_id>` returns `TrialDetails` (with `evaluations`); `load_model` round-trips a packaged model.
- [x] **A4.3 — summary + queries.** `experiment summary`; `recent_failures` / `strategies_attempted` (no `training_origin` filter) compose over the seam.
- [x] **A4.4 — cleanup cascade.** `experiment delete <id> --apply` removes that experiment's MLflow + GCS + local-dir blobs in **one** universe; soft-delete default, `--hard-delete` runs `mlflow gc`; idempotent re-run; never crosses mode/namespace.

Phase 4 gate evidence (2026-05-28): `uv run pytest tests/unit tests/contracts tests/integration -v` -> `222 passed`; `AUTOML_PHASE4_E2E=1 MLFLOW_TRACKING_URI=http://127.0.0.1:54321 uv run pytest tests/e2e/test_phase4_experiment_trial_cleanup.py -v` -> `1 passed`; `AUTOML_PHASE3_E2E=1 MLFLOW_TRACKING_URI=http://127.0.0.1:54321 uv run pytest tests/e2e/test_phase3_eval_breadth.py -v` -> `1 passed`; `uv run pytest tests/contracts -v` -> `9 passed`. Import ratchets found no new `automl_legacy` imports and no PyPI `mlflow` imports outside `automl/mlflow/**`.

## Phase 5 — agent domain + hooks

- [x] **A5.1 — proposer context.** `experiment proposer-context` (→ `gather_proposer_context`) emits the dict packet (views + trial reads + data seam; `find_prior_experiment` cold-start by creation_time).
- [x] **A5.2 — Proposal contract.** `validate proposal` accepts/rejects against the `Proposal` dataclass roster + `DISALLOWED`; `--output` writes validated JSON on pass.
- [x] **A5.3 — full loop.** `experiment run` launches the proposer→coder loop; it emits a validated `Proposal`, runs a trial, and `agent/timeline` reconciles hook events into MLflow (seam-routed; agent writes its own `agent/manifest.json`).

Phase 5 gate evidence (2026-05-28): `uv run pytest tests/unit tests/contracts tests/integration -v` -> `244 passed`; `AUTOML_PHASE5_E2E=1 MLFLOW_TRACKING_URI=http://127.0.0.1:54321 uv run pytest tests/e2e/test_phase5_agent_hooks.py -v` -> `1 passed`; `AUTOML_PHASE4_E2E=1 MLFLOW_TRACKING_URI=http://127.0.0.1:54321 uv run pytest tests/e2e/test_phase4_experiment_trial_cleanup.py -v` -> `1 passed`; `AUTOML_PHASE3_E2E=1 MLFLOW_TRACKING_URI=http://127.0.0.1:54321 uv run pytest tests/e2e/test_phase3_eval_breadth.py -v` -> `1 passed`; targeted Phase 5/code-review sweep `uv run pytest tests/unit/agent tests/unit/validate tests/unit/cli tests/contracts tests/e2e/test_phase5_agent_hooks.py -v` -> `41 passed, 1 skipped`; `uv run ruff check automl hooks tests/unit/agent tests/unit/cli tests/unit/validate/test_proposal_validation.py tests/unit/validate/test_model_validation.py tests/e2e/test_phase5_agent_hooks.py` -> `All checks passed`; import ratchets found no new `automl_legacy` imports and no PyPI `mlflow` imports outside `automl/mlflow/**`.

## Phase 6 — surface & isolation breadth

- [x] **A6.1 — CLI catalog complete.** Every verb in 00 §11.1 resolves to its library destination; `--json` works where specified. Phase 6 implements the noun-first dispatcher/modules for `project`, `experiment`, `trial`, `data`, `eval`, and `validate`, with top-level session flags.
- [x] **A6.2 — `--dry-run` universe.** `automl --dry-run experiment run` routes to the `dry_run/` universe (separate MLflow names + GCS + local dirs); cleanup of it leaves real untouched.
- [x] **A6.3 — `--namespace` full-universe isolation.** `automl --namespace qa …` runs a full-fidelity trial in a `qa/` sandbox (MLflow + GCS + local dirs all prefixed); `automl --namespace qa experiment delete <id> --apply` cleans only `qa`, never real (`""`); composes with `--dry-run` (`qa/dry_run/…`).
- [x] **A6.4 — plugin/skills resolve.** `render_context.py` verb renames + dropped flags applied; `model-contract.md` documents the required-preprocessing contract; all skill commands hit new verbs.

Phase 6 gate evidence (2026-05-28): targeted Phase 6/code-review sweep
`uv run pytest tests/unit/cli tests/unit/project tests/unit/validate tests/unit/eval tests/unit/agent/test_launch.py tests/unit/runner/test_session_lock.py tests/contracts/test_phase6_skill_commands.py tests/integration/cleanup/test_experiment_delete.py tests/e2e/test_phase6_surface_isolation.py -v` -> `108 passed, 1 skipped, 2 warnings`;
`uv run ruff check automl hooks tests/unit/agent tests/unit/cli tests/unit/project tests/unit/validate tests/unit/runner/test_session_lock.py tests/contracts/test_phase6_skill_commands.py tests/e2e/test_phase6_surface_isolation.py` -> `All checks passed`;
`uv run pytest tests/contracts -v` -> `11 passed`;
`uv run pytest tests/unit tests/contracts tests/integration -v` -> `273 passed, 2 warnings`;
`AUTOML_PHASE6_E2E=1 MLFLOW_TRACKING_URI=http://127.0.0.1:54321 uv run pytest tests/e2e/test_phase6_surface_isolation.py -v` -> `1 passed`;
Phase 5/4/3 external preservation gates each -> `1 passed`.
Import ratchets found no new `automl_legacy` imports and no PyPI `mlflow` imports outside
`automl/mlflow/**`.

## Phase 7 — cutover

- [x] **A7.1 — checklists green.** `migration-checklist.md`: zero un-dispositioned rows. This file: all behavior rows `[x]`.
- [x] **A7.2 — test prune complete.** Per-domain unit tests rebuilt (TDD); legacy test debt shed; tiers (unit/integration/contract/e2e) populated; `tests/contracts/test_pytest_structure.py` updated.
- [x] **A7.3 — full e2e.** A complete agent-loop run against the harness (propose → implement → eval → leaderboard) passes end-to-end on the new tree.
- [x] **A7.4 — cutover.** `git rm -r automl_legacy/ tests_legacy/`; the new `automl/` package is the package; legacy gone.

Phase 7 gate evidence (2026-05-28): targeted Phase 7 gate
`uv run pytest tests/unit/trial tests/unit/runner tests/unit/cli tests/contracts tests/integration/runner tests/e2e/test_phase7_cutover.py -v` -> `58 passed, 1 skipped, 3 warnings`;
`uv run pytest tests/unit tests/contracts tests/integration -v` -> `283 passed, 2 warnings`;
`uv run pytest -v` -> `283 passed, 7 skipped, 2 warnings`;
`uv run pytest tests/contracts -v` -> `13 passed`;
`uv run ruff check automl hooks projects/payment_routing/config.py projects/example_homecredit/config.py projects/example_homecredit/model tests/unit/trial tests/unit/runner/test_trial_folder_execution.py tests/unit/cli/test_phase6_cli_catalog.py tests/unit/mlflow/test_trial_artifacts.py tests/contracts tests/e2e/test_phase7_cutover.py` -> `All checks passed`;
`AUTOML_PHASE7_E2E=1 MLFLOW_TRACKING_URI=http://127.0.0.1:54321 uv run pytest tests/e2e/test_phase7_cutover.py -v` -> `1 passed`;
Phase 6/5/4/3 external preservation gates each -> `1 passed`. Import ratchets found no new
`automl_legacy` imports and no PyPI `mlflow` imports outside `automl/mlflow/**`; the remaining
`automl_legacy`/`tests_legacy` Python hits are contract-test string literals.

## Final whole-refactor audit

- [x] **Spec alignment review.** Re-read the approved specs/index, implementation strategy,
  acceptance checklist, and migration checklist against the finished tree. Clear drift fixed in
  docs and active skill/reference surfaces.
- [x] **Migration completeness review.** `migration-checklist.md` has zero `[ ]`, `[/]`, or `[?]`
  implementation rows; audit-note concerns are closed or intentionally dropped.
- [x] **Architecture and safety review.** Contracts/import ratchets pass; active surfaces no
  longer mention retired command/env/snapshot-era names; no `automl_legacy` tree or imports; only
  `automl/mlflow/client.py` imports PyPI `mlflow`.
- [x] **End-to-end behavior review.** Full local suite and required external Phase 7/6/5/4/3
  gates passed with the worktree `.env` loaded and
  `MLFLOW_TRACKING_URI=http://127.0.0.1:54321`.
- [x] **Documentation review.** Root README, refactor README, plan README, implementation
  strategy, acceptance checklist, migration checklist, Phase 7 plan, skills, agents, and
  references reflect the cutover state. A later follow-up coverage/readiness review recorded
  deferred/spec-gap items in `spec-coverage-review.md` and `final-review-open-items.md`; those
  items need triage before merge readiness can be claimed.

Final audit evidence (2026-05-28):
`uv run pytest -v` -> `283 passed, 7 skipped, 2 warnings`;
`uv run pytest tests/contracts tests/integration/cleanup/test_experiment_delete.py -v` ->
`16 passed, 2 warnings`;
`uv run ruff check automl hooks skills agents references tests projects/payment_routing/config.py projects/example_homecredit/config.py projects/example_homecredit/model` ->
`All checks passed`;
`git diff --check` -> clean;
`AUTOML_PHASE7_E2E=1 MLFLOW_TRACKING_URI=http://127.0.0.1:54321 uv run pytest tests/e2e/test_phase7_cutover.py -v` ->
`1 passed, 27 warnings`;
`AUTOML_PHASE6_E2E=1 MLFLOW_TRACKING_URI=http://127.0.0.1:54321 uv run pytest tests/e2e/test_phase6_surface_isolation.py -v` ->
`1 passed, 29 warnings`;
`AUTOML_PHASE5_E2E=1 MLFLOW_TRACKING_URI=http://127.0.0.1:54321 uv run pytest tests/e2e/test_phase5_agent_hooks.py -v` ->
`1 passed, 31 warnings`;
`AUTOML_PHASE4_E2E=1 MLFLOW_TRACKING_URI=http://127.0.0.1:54321 uv run pytest tests/e2e/test_phase4_experiment_trial_cleanup.py -v` ->
`1 passed, 26 warnings`;
`AUTOML_PHASE3_E2E=1 MLFLOW_TRACKING_URI=http://127.0.0.1:54321 uv run pytest tests/e2e/test_phase3_eval_breadth.py -v` ->
`1 passed, 19 warnings`.
