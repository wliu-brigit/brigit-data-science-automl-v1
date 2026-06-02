# AutoML Refactor — Design specs (`spec/`)

> 📍 **Front door is [`../README.md`](../README.md)** — go there for current status, the doc map,
> and "what to do next." **This `spec/` folder is the design record** (*what & why*); the
> **execution** material (strategy, checklists, per-phase plans) lives in [`../plan/`](../plan/).
> The rest of this file is the design index + the per-sub-spec "what's done" detail.

This folder holds the design artifacts for the **brigit-automl** library refactor.
The refactor rebuilt the library under the new four-layer architecture with six canonical
noun domains. Phase 7 removed the frozen `automl_legacy/` tree; the fresh `automl/`
package is now the only package in the `automl_dev-refactor` worktree.

**Started:** 2026-05-19
**Design status:** **all sub-specs 00–11 approved; final cross-doc carry-back/consistency pass COMPLETE (2026-05-27).** The old `experiment/` mega-domain was split into three peers (`experiment/` + `trial/` + `agent/`), so sub-spec 09 became 09/10/11. All open `🟡` items applied/deferred; the pass also caught + fixed **nine latent carry-back gaps** (see `open-questions.md` → "FINAL-PASS CLOSEOUT"). **Design is frozen; Phases 0–7 are implemented and Phase 7 performed the hard cutover. Build status and audit closeout live in [`../README.md`](../README.md) and [`../plan/`](../plan/).**
**Workspace root:** `/Users/zhengisamazing/1.python_dir/brigit/`

---

## Starting a fresh session → use the front door

**Resuming the refactor? Read [`../README.md`](../README.md), not this file.** The front door
owns status + the next action + the full doc map. This `spec/` README is the **design index**
(read it when you need design *detail* on a domain). Design is **frozen**; the work now is the
**build** (tracked in [`../plan/`](../plan/)) — so the old "continue with the next sub-spec"
guidance no longer applies.

---

## How to read this folder (design detail)

Read in this order on a fresh session:

1. **This README** — orientation
2. **`00-structural-design.md`** — the parent spec. Vocabulary, four-layer architecture, domains, folder shape, decision rules. Required.
3. **`../plan/migration-checklist.md`** — the audit doc. Lists every public symbol in the legacy library and its new home. Status column shows what's done.
4. **`open-questions.md`** — closed design-decision record for ambiguities surfaced during sub-spec work.
5. **Sub-specs (`01-*.md`, `02-*.md`, …)** in order — each settles the interface for one cross-cutting concern or domain. Open the ones that are done; skip ones that don't exist yet.

---

## Document types

The folder contains four kinds of docs. Knowing which is which prevents confusion:

| Type | File pattern | What it is | Mutability |
|---|---|---|---|
| **Structural design** | `00-structural-design.md` | The parent design — vocabulary, layers, domain set, folder shape, cross-cutting rules. Source of truth for *structure*. | Updated when sub-specs surface structural changes. Stable otherwise. |
| **Sub-specs** | `NN-<topic>.md` for `NN ≥ 01` | Per-topic interface designs. Each settles function signatures, return types, extension hooks for one concern. | Frozen once approved. |
| **Migration checklist** | `../plan/migration-checklist.md` | Rolling audit — every public symbol in the legacy library, mapped to its new home, with Status column. | Updated as migration progresses. |
| **Open questions** | `open-questions.md` | Closed design-decision record for ambiguities found during design. | Historical; reopen only for explicit design corrections. |

**Don't confuse them.** If a sub-spec contradicts the structural spec, the structural spec wins until consciously updated. If the checklist disagrees with a sub-spec, the sub-spec wins for design intent, while `../plan/` wins for execution status. Specs > plan for design; plan > specs for live progress.

---

## Current state

### What's done

- **`00-structural-design.md`** — approved. Defines:
  - Six canonical nouns: Project, Dataset, Experiment, Trial, Proposal, Model
  - Four-layer architecture: Surface / Domain / Framework / Utility
  - Six top-level domain folders: `project/`, `data/`, `model/`, `eval/`, `runner/`, `experiment/`
  - Framework: `mlflow/`, `validate/`
  - Utility: `utils/`
  - Surface: `cli/` + skills + `hooks/`
  - CLI verb sketch + principles

- **`01-project-context.md`** — approved. Defines:
  - Three names: `config.py` (file) → `ProjectConfig` (loaded) → `Session` (active)
  - Entry point: `automl.use_project(name, ...)`
  - Access: `automl.session()`, `automl.active_session()` (cm), `automl.update_session(**kwargs)`, `automl.clear_session()`
  - None-semantics for unfilled config fields (no exploration flag)
  - Per-function signature convention: `session: Session | None = None`, resolved via `session if session is not None else automl.session()`
  - Eight-step `ProjectConfig.load()` contract
  - CLI overrides layer onto Session, not ProjectConfig
  - Async-safety guidance: `active_session()` is async-safe; `use_project()` and `update_session()` are process-level only
  - **`_bind_mlflow_for(session)` helper (§4.6):** sub-spec 02 carry-back applied directly. `use_project`, `update_session`, and `active_session` all call this helper in lock-step so the mlflow persistence layer's bind state always tracks the active session atomically.

- **`05-data.md`** — approved 2026-05-24 (three-agent review + holistic cross-doc consistency pass). Defines:
  - **Vocabulary (Q1):** "snapshot" fully retired at code level in `data/` → "dataset" (classes, fields, files, GCS paths, MLflow tags). Clean cut, no back-compat.
  - **Two Tier 3 anchors (Q2):** `DataSource` (source-side) + `DataPipeline` (orchestration-side); composition via `DataSpec.pipeline_cls`; `data/adapters/` legacy wrappers deleted.
  - **Verbs + split-at-load (Q3):** `build_dataset` / `materialize` / `list_datasets` / `load_dataset(_by_id/_by_trial)`. Materialized Dataset = full df + registry; **no train/test persisted** — splits applied at LOAD time via pyarrow push-down (leakage-safe). `load_dataset_by_trial` resolves splits from the trial contract, not config.py.
  - **Typed objects (Q4):** `Dataset` / `LoadedDataset` / `LoadedSlice` / `DatasetIndex` + `ComponentHashes`. Composition over flat. Forward-only field audit dropped `prepare_event_id` / `run_mode` / `experiment_id`-on-Dataset / `source_event` / `gcs_base_path`-as-field.
  - **Profile (Q5):** single `data/profile.py` (not a folder); pluggable named-function check/chart lists; MLflow writing → `mlflow/project/artifacts.py` (§9.1). Profiles move to project-overview run (bug-fix).
  - **FeatureRegistry (Q6):** lift `core/feature_registry.py` → `data/features.py`; drop golden/weak learning; add `derived` + `source_columns` + `add_derived()`. Feature-store research → description/tags/ownership/etc. rejected for our context.
  - **DataSpec + DataPipeline (Q7):** thresholds/dry_run_rows solely on DataSpec; ctor 14→3 args (`spec`, `session`, `refresh_source`); `constant_drop_threshold` default 0.99→1.0 (was disabling the check).
  - **Splits (Q8):** free-form named dict replaces hardcoded train/test `Split`. **Lives in `project/run_config.py`** (not data/ — dependency-direction). Half-open ranges, no overlap; pyarrow filter built reader-side in `data/registry.py`.
  - **TrialDataContract (Q9):** `RunDataContract`→`TrialDataContract`; four types; any-named-slices; four integrity validators (L1 contract↔Dataset, L2 loaded↔manifest, L3 slice↔contract-hash, L4 contract↔trial-tags).
  - **Three review reversals (00 won over draft):** hash primitives → `utils/hashing.py` PUBLIC (§13.8); CLI verb `automl data profile` (§11.1); `Splits` in `project/`. Plus a pre-existing 00↔02 mlflow folder-layout inconsistency fixed in the same pass.
  - **Carry-backs APPLIED** to 00 (§5/§5.1/§7/§8.1/§8.2/§13.8/§719/appendix), 01 (import line + DataPipeline subclass note), 02 (§4/§6.1/§6.2.3/§6.3.4/§9). Migration-checklist + open-questions updated. Two items carried to sub-spec 07 (eval pre-flight gate, `of_data_snapshot_id` naming).

- **`04-validate.md`** — approved 2026-05-23 (two rounds of three-agent review). Defines:
  - `validate/` framework: four files (`base.py`, `targets.py`, `synthetic.py`, `__init__.py`); `registry.py` file deleted (surviving pieces — `_import_project_validators` + `_RESET_FOR_TESTS` — moved to `validate/targets.py`).
  - **Three orchestrators only**: `project`, `model`, `proposal` (drop `config`, `contracts`; no speculative `experiment`). `Target` literal pruned from 5 entries to 3.
  - **Built-in checks are direct function calls** — no `@register` decorator, no `_CHECKS` global, no `inspect.signature` filtering, no `register_all()` defensive helper. `CheckSpec` dataclass deleted. **Per-check exception-wrapping preserved** via a `_safe(name, fn, **kwargs)` helper — a crashing check emits an `Issue("error", "<name>.crashed", ...)` rather than taking down the whole orchestrator.
  - **Project-side extension**: `projects/<name>/validators.py` exports `PROJECT_CHECKS = {"project": [fn]}`. **`"project"` target only** — `"model"` and `"proposal"` dropped per `feedback_extension_points_follow_demand` (no real consumer, can re-add). Pinned signature `fn(*, session: Session) -> Iterable[Issue]` (aligns with sub-spec 01's `session` convention).
  - **`validate.model(cls, *, df, registry) -> ValidationReport`** — caller builds the sample. Runner pulls forward **only the 200-row pre-fit sample** (not the full snapshot — the earlier wording would have lost today's MLflow-capture of main-fit load failures). Phase ordering: load pre-fit sample → pre-fit → open MLflow run → load full snapshot + fit + eval + log. Observability semantics preserved.
  - **`ModelProbe` dataclass deleted** — replaced by private `_try_fit(cls, df, registry) -> (instance, error, error_stage)` inside the orchestrator. `sample_kind` parameter dropped (error messages carry row count + exception type already).
  - **Proposal target collapses with `propose.validate()`** — three duplicate Issue/ValidationReport types unify in `validate/base.py`. `agent/checks.py::proposal_schema` (moved from `experiment/checks.py` in the 09 decomposition) absorbs today's `propose.validate()` logic + the adapter in `proposal_checks.py`. Three callers updated (`cli/propose.py`, `cli/trial.py:76`, the adapter). **`--output <path>` flag preserved** by adding it to `automl validate proposal` (keeps `skills/automl/scripts/render_context.py::safe_commands.persist_proposal` working).
  - **`ValidationReport` gains `schema_version: int = 1` + `from_dict`** per sub-spec 02's typed-schema pattern (it's persisted as a trial artifact).
  - `Severity = Literal["error", "warning"]` (drop `"info"`); `ValidationError` + `ValidationReport.raise_if_failed()` deleted (no callers).
  - **Pinned `_load_project_module(session, module_name)` helper** lives at `project/_imports.py`; used by data + eval + project checks.
  - **Carry-backs applied to parent specs:** §11.1 (CLI verb catalog — `validate` lists only project/model/proposal; `validate proposal` accepts `--output`), §13.1 (CheckSpec deleted from schema table; `validate/types.py` → `validate/base.py` file rename; `ValidationReport` gains `schema_version`+`from_dict`), §15.2 ("Validate registry import timing" deferred item removed), §17.8 ("registry behavior is unchanged" wording updated — the registry is gone).

- **`03-cleanup.md`** — approved 2026-05-22. Defines:
  - Three sibling delete verbs (`project delete` / `experiment delete` / `trial delete`) replacing legacy single `cleanup --scope` verb
  - Cascade engine in `project/cleanup.py`; thin per-noun wrappers in `experiment/cleanup.py` + `trial/cleanup.py` (was `experiment/trial/cleanup.py` before the 09 decomposition)
  - Layer order: MLflow → GCS → local (inverse of writer's GCS-then-MLflow contract)
  - Soft delete is the default; `--hard-delete` opt-in runs `mlflow gc` against the backend store
  - Plan/apply two-phase (preview by default; `--apply` to commit; no `--yes` flag)
  - dry_run is a top-level CLI flag (session-wide), NOT a per-verb argument — mode comes from `session.dry_run`
  - Trial-delete enforces session/run mode consistency via MLflow lookup (`get_run` → `get_experiment` → parse name → verify project + mode against session)
  - Continue-and-collect error model (per-target status in `CleanupResult`); idempotent re-runs
  - Plan enumeration uses `view_type=ACTIVE_ONLY` (already-soft-deleted records stay deleted)
  - Strict-isolation principle extended: trial sandbox dirs path-segregated by mode (carry-back to sub-spec 08)
  - Typed schemas: `CleanupPlan` / `CleanupResult` / `CleanupReport` with `schema_version: int = 1` + `from_dict`
  - **CLI-shape carry-back applied to structural spec §11.1** — noun-first verbs everywhere except `validate`; top-level `--dry-run`; 21-verb catalog (was ~25 sketched verbs with inconsistencies)
  - **`mlflow_artifacts_destination` carry-back applied to sub-spec 01 §3.1** — added to `ProjectConfig` for `mlflow gc` parameter threading

- **`02-mlflow-seam.md`** — approved. Defines:
  - Connection state bound once via `mlflow.bind(...)`; no `Route`; identifier per-level
  - Per-noun folders: `mlflow/project/`, `mlflow/experiment/`, `mlflow/trial/` with internal sub-modules
  - `StorageError` single exception type (wraps backend errors via `__cause__`)
  - GCS-then-MLflow writer ordering; shared `_atomic.py` partial-write helper
  - Centralized tag-key constants in `mlflow/tags.py`
  - Active-run context manager yields `run_id: str` (no `ActiveTrial` class); same `log_*` API for active + post-hoc
  - Two-tier artifacts (loose `log_json` + typed writers with schema)
  - Additive-only schemas with `schema_version: int = 1` placeholder; no version-dispatch until needed
  - Multi-instance eval/predictions with `label` parameter
  - Level contract table (project / experiment / trial — what lives where)
  - Foundation surface only; deferred analytics get **no placeholder file** (sub-spec 09 §Q4 — `recent_failures`/`strategies_attempted`/`compare` are in-scope view helpers; only `runs_using_strategy`/`runs_in_metric_band` deferred)

- **`06-model.md`** — approved 2026-05-24 (design interview Q1–Q7 + mechanical pass + three-agent review + fixes applied). Defines:
  - **Reframe:** model = **preprocessing → estimator**; preprocessing becomes a co-equal, formally-contracted part of the domain (not just the estimator).
  - **New feature — project-mandated preprocessing (Q1 C):** a project can declare mandatory preprocessing (e.g. a WOE encoder on a categorical risk column) every trial model must use, enforced in validation + surfaced in the coder prompt. Contract-level extension point owned by the model domain — NOT framework auto-injection (author must *actively* use it).
  - **Unit (Q2 A):** typed `RequiredTransformer(name, transformer, input_cols)` in `model/preprocessing.py`; column-scoped, sklearn-native, splices into `ColumnTransformer`.
  - **Enforcement (Q3 B):** inspection gate, framework-owned. Project declares *data* in `config.py`; `BaseModel.required_transformer_entries(session=None)` hook loads canonical `clone`d entries; framework check `check_required_transformers` in `validate.model` enforces. Empty-default = no-op (existing models + Home Credit harness unaffected). The targeted re-add of sub-spec 04's deferred `"model"` check.
  - **Structural mandate (Q4 A):** when requirements exist, `self.preprocessor` must *be* a top-level `ColumnTransformer` (not a Pipeline wrapping one — S2); downstream steps live in `self.model`. Gate reads fitted `transformers_` triples (C1).
  - **Declaration (Q5):** `config.py` declares `REQUIRED_TRANSFORMERS`; classes live in `projects/<name>/model/preprocessing.py` (mirror-core); serialization verified via `code_bundle` (cloudpickle by-reference + bundled `projects/` tree).
  - **Integrity (Q6 i):** type + columns floor; no hyperparameter pinning.
  - **Prompt surfacing (Q7 A):** `describe_required_transformers(session)` helper → `TrialProposal.required_preprocessing` field (proposer writes, coder reads); single enforcement point = the model gate.
  - **Migration corrections:** `package_model` (notebook→source) → `trial/packaging.py` and `load_model(run_id)` (mlflow pyfunc) → `trial/show.py` — both move OUT of `model/` (model domain deps = `errors` only). Land in **sub-spec 10 (`trial/`)** post-decomposition. `model/packaging.py` holds only `save_model`.
  - **Carry-backs APPLIED:** sub-spec 01 (`ProjectConfig.required_transformers`), structural §8.3+§7 (`model/preprocessing.py`), sub-spec 04 (gate + ambient-session). **Carry-forwards:** **sub-spec 11 (`agent/Proposal.required_preprocessing`)** (post-decomposition; was "sub-spec 09 / TrialProposal"), plugin layer (agent wiring + `model-contract.md`). **Home Credit deliverable:** real `WOEEncoder` on `ORGANIZATION_TYPE`.

- **`07-eval.md`** — approved 2026-05-24 (design interview Q1–Q6 + three-agent review + fixes applied). Defines:
  - **Unification checkpoint (Q1):** keep both `EvalDataset` kinds, but `split_view` **delegates** bucket realization to data's `load_dataset_by_id(of_dataset_id, split_range=)` — removes the triplicated `_realize_split_view_frame` (the one landed trigger). Full substrate+lineage+role unification recorded as **north-star** in §13.8, re-opened at the `eval → data` seam-thickness tripwire / a 3rd byte-owning artifact family. Not a no-op, not a rebuild — the lighter slice.
  - **Recipe-only split_view identity (Q2):** `eval_dataset_id` recipe-derived; drop realized `schema_hash`/`content_hash` from the manifest (load *and* publish); integrity from the **content-addressed `of_dataset_id` + data's L2 load-time validation** (NOT L3/`verify_loaded_slice`, which is trial-contract-scoped). Empty-bucket detection moves publish→first-load (intentional).
  - **Full "snapshot" retirement (Q3):** clean cut in `eval/` — every symbol/path/tag; closes carry-back `of_data_snapshot_id → of_dataset_id`.
  - **Pre-flight gate (Q4):** two checks — early (surface-layer data-build verb vs. materialized schema, built-or-reused) + pre-fit (runner vs. loaded frame); one pure predicate `missing_eval_columns` in `eval/checks.py`; the `data → eval` back-edge deleted. Confirms §04's three-layer validator model (no registry route).
  - **Verb surface (Q5):** two entry points; `eval/runner.py` deleted; `run()` → `evaluate_frame()`; `session` convention; `_model`/`_model_feature_registry` injection preserved.
  - **Type consolidation (Q6):** mirror data's vocabulary — four types removed (`EvalSnapshotPointer`, `AugmentationPointer`, the separate manifest schema, and `EvaluateResult`+`EvalResults` → singular **`EvalResult`**). `EvalIndex`/`Predictions` typed; `EvalDataset`/`Augmentation` absorb their pointers (paths as properties); `mlflow_url` dropped (CLI derives via mlflow-seam helper); `cached` the lone runtime-only field.
  - **Carry-backs APPLIED:** §00 §8.4 (`load_dataset_by_id` in eval's allowed imports + tripwire), §00 §13.8 (identity reshape + unification resolution), §00 Tier-2/schema tables (`EvalResults`→`EvalResult`), sub-spec 02 (`EvalResults`/`EvalResultsRef`→`EvalResult`/`EvalResultRef`; `eval_snapshot_id`→`eval_dataset_id` in the mlflow-seam eval writer/payload/listing). **Carried forward:** sub-spec 05 (`load_dataset_by_id` must run L2 by default + accept multiple disjoint bucket ranges), sub-spec 01 (`project → eval` edge for `evaluation_spec`).

- **`08-runner.md`** — approved 2026-05-25 (design interview Q1–Q6 + three-agent review + fixes applied). Defines:
  - **Scope = Tier 1:** comply with all carry-backs + mechanical hygiene; **no** stage-pipeline / shared-state-carrier / pluggable runner. The monolith's coupling (one global failure boundary in the ~250-line `finally`) is intrinsic; the modular "stage" runner is the **deferred Tier-2 north-star**, re-opened only when a real 2nd runner shape (HPO/ablation/distributed) appears — matching 00 §13.3/§17.
  - **Q1 trial folder:** mode-segregated path `projects/<project>/experiments/[dry_run/]<project>/<experiment_id>/<slug>/`; `trial.create` builds it (was `experiment.trial.create` pre-decomposition), the **runner verifies** it (universe-isolation guard, upgrading today's weak containment check); path helper in `runner/`. Corrects 03's "runner creates dirs" wording.
  - **Q2 identity:** exec-time number assignment kept (`trial_id = <number>_<slug>`); the MLflow `next_trial_number` query moves to the seam (carry-back to 02) to break the backward `runner→experiment` import.
  - **Q3 phase order:** 04's `sample → prefit → open run → full-load → fit → eval → log` re-expressed in 05's `load_dataset` vocab; pre-fit sample = `load_dataset(split_name=train_split).df.head(200)` (no new loader API); `session.dry_run`; SIGALRM armed before phase 1.
  - **Q4 data contract:** **fit-slice-only** `TrialDataContract`, built from `LoadedSlice` + written via `mlflow.trial.artifacts.write_trial_data_contract`; new per-slice tags. **`TrialRef` keeps both `trial_id` + `run_id`** — carry-back to 05 (the "drop as duplicate" misread 00 §5's vocabulary as field-dedup; they're distinct strings).
  - **Q5 eval:** runner=fit-only, `evaluate() -> EvalResult` owns eval-data loading; pre-fit `EvalSpec.validate_columns` on the **fit frame** (split_view; external eval checked inside `evaluate()`); train-eval diagnostic kept; validation fixture rebuilt from a fit-frame sample. 06's required-transformer gate needs no runner change.
  - **Q6 decomposition:** `runner/trial.py` (the chain) + cohesive modules (`paths`/`contract`/`validation`/`_pyfunc_check`/`manifest`/`_modules`/`session_lock`); **no `runner/stages/`** (carry-back to 00 — appendix migration line ~840 only); dead code (`_write_error_log` shim / `_load_trial_model_module` / `_frame_shape`) + star-import removed; the 286-line inline pyfunc round-trip extracted to `_pyfunc_check.py`; session lock moves in (CLI `automl trial lock` → `runner.trial_lock`).
  - **Eval-slice lineage** in the trial contract: fit-only **accepted as intentional** (splits ranges + eval-domain integrity cover it).
  - **Carry-backs APPLIED:** 03 §3.4 (path wording), 05 Q9 (`TrialRef.run_id`), 02 (`next_trial_number`), 00 (drop `runner/stages/` from Tier-1 target). Migration-checklist + open-questions updated. Open item carried to 05/09: `RunDataContract.to_split_view()` consumers.

- **`09-experiment.md`** — approved 2026-05-25. The slimmed `experiment/` (Experiment noun + overview state + cross-trial views). `Experiment = ExperimentOverview` (one type, facade-aliased); `lifecycle.create` + lazy ensure both kept (no predecessor param — dead tag); raw searches at the seam, `recent_failures`/`strategies_attempted`/`compare` in-scope view helpers, `runs_using_strategy`/`runs_in_metric_band` deferred (zero-file); drop public `experiment_id()`; typed `LeaderboardData`/`ComparisonResult`, `summary` stays a dict, `learning_counts` dropped. Split the old `experiment/` mega-domain into `experiment/`+`trial/`+`agent/` (swept across 00/02–08 + living docs). §18 carry-backs to 00/02 batched with sub-spec 10's closeout.

- **`10-trial.md`** — approved 2026-05-26 (design interview Q1–Q10 + three-agent review + fixes). The `trial/` domain (Trial noun, top-level peer). Defines:
  - **Read types (Q1/Q2):** `show_trial -> TrialDetails` (seam `get_details` maps run state + `evaluations=None`; `show_trial` fills `evaluations: list[EvalResult] | None`); `TrialSummary` and `TrialDetails` **independent** (no composition) with shared private seam builder helpers; `ComparisonResult.runs: list[TrialDetails]` (resolves 09's deferral); url derived at boundary.
  - **Field reconciliation (Q3):** five additive fields onto `TrialSummary` (`trial_number`/`hypothesis`/`training_origin`/`training_time_s`/`n_features`) — carry-back to 02; legacy 19-key dict otherwise covered/raw/derived/dropped.
  - **Write schemas (Q5–Q7):** `TrialMetadata` (`run_mode`/`dry_run` dropped — universe is path+session); `SeedSelection` + typed `ModelSource`; `TimingReport`; **slim `TrialManifest`** TOC + `run_id` (the legacy navigation spine is write-mostly — dropped, carry-back to 08).
  - **Mechanics (Q8–Q10):** session convention sweep (no project/project_root/dry_run); `run.py` dropped; `load_data_snapshot`→`data`; **zero-file `checks.py`** (Q9); per-operation files + `types.py`(read)/`metadata.py`(write) split; `package_model`/`load_model` land here (06 correction).
  - **Carry-backs APPLIED:** 02 (`TrialSummary` +5 fields, `TrialDetails` fields defined), 09 (`ComparisonResult.runs`), 00 §7/§8.7 (eval dep + checks.py drop + exports), 08 (manifest writer slims), checklist (`SLUG_RE`→`utils/` to avoid `trial→agent` cycle). Open-questions + migration-checklist updated.

- **`11-agent.md`** — approved 2026-05-27 (design interview + three-agent review + fixes). The `agent/` domain (the agentic loop, **relocate-only**; 00 §8.8 + §17.11). Defines:
  - **Proposal contract (Q1/Q2):** `propose/` → typed frozen `Proposal` dataclass in `agent/proposal.py` (was no class — a raw dict); `schema_version` stays 2; `from_dict`/`to_dict`; the **dataclass is the single roster source** (check introspects `dataclasses.fields`), `DISALLOWED=("parent_id",)` the lone constant, format rules (slug regex / `seed_hint` enum / non-empty-list) in the check. `required_preprocessing: list[dict] | None` (06 carry-in — allowed-not-enforced; single gate = model gate). `SLUG_RE` from `utils/` (10; kept there after review — generic snake_case primitive shared by `agent/` + `trial/`).
  - **`proposal_schema` (Q3):** `agent/checks.py::proposal_schema(proposal: dict, *, session=None) -> list[Issue]` (canonical `validate.Issue`, 04); **session-resolves the allow-list** via `project.dependencies` — drops the explicit param + the two `--allowed-deps` CLI flags and **fixes the `cli/trial.py` allow-list tautology**. `automl validate proposal` keeps `--json`+`--output` (04).
  - **`proposer_context` (Q4/Q5):** `get_context` (~230L) + `proposer_packet` rebuilt as `agent/proposer_context.py::gather_proposer_context` — a **dict composer** (no raw MLflow searches) over `experiment.views` (09) + `trial` reads (10) + the data seam + `find_prior_experiment` (agent-owned; cold-start; **creation_time** ordering — cheap win). Drops: `top_trials` dup, project/experiment **learnings** (out of scope), `artifact_uris`/`artifact_errors`, `primary_eval` per-row enrichment. `data_context` reshaped to 05 vocab (`active_dataset`, `dataset_usage`, profile per 05 Q5).
  - **Metric ranking reconciliation:** the sort metric is a **parameter defaulting to `config.primary_metric`** (the experiment's *current* primary, mutable; optional `--metric` override); each trial's own primary = **provenance only**; trials missing the current metric are **reported** (*"x/n not scored on `<metric>`"*) — **no re-eval hook** (§17.5 stays forward-looking).
  - **Launcher (Q6):** `cli/run_loop.py` → `agent/launch.py::build_launch(*, session=None, …) -> LaunchSpec` (session model routing from `config.models`); pure builder, CLI `experiment run` executes; `LaunchSpec`/`ClaudeRole` + `agents/*.md` parsing ported; injects `AUTOML_INHERIT_DRY_RUN`.
  - **Timeline (Q7–Q9):** `hooks/agent_timeline.py` (1955L) → `agent/timeline.py` (`handle_event` + `publish`; reconciliation **ported verbatim** as internal; `summarize` subcommand + `publish_mlflow` param dropped) + thin `hooks/` stub. Writes **seam-routed** (`mlflow.trial`/`experiment.log_json`/`log_metric`); GCS via `utils.io.gcs` + `mlflow/_routing.py` (drops the `importlib` `gcs_paths` hack + the manifest-merge — agent writes its own `agent/manifest.json`). Route + dry_run from session; `AUTOML_DRY_RUN` → **`AUTOML_INHERIT_DRY_RUN`** (transport-only); kills the `sys.argv` parse / route-string parsing / lock-lookup / `route_namespace`. `timeline.py` stays one file (relocate-verbatim).
  - **Carry-backs RECORDED (7):** 00 (§8.8 add `publish`; §11.1 validate-targets 6→3 — stale 04), 09 (leaderboard default metric + `LeaderboardData` unscored-count + preserve `strategies_attempted` no-origin-filter), 02 (`top_n_by_metric` cross-trial-stable sort key + `_routing.py` agent-events prefix helper), 07/08 (confirm namespaced metric logging), 08+cross-cutting (`AUTOML_DRY_RUN`→`AUTOML_INHERIT_DRY_RUN` + drop the obsolete metadata-conflict check). Plus plugin-layer carry-forwards (skill `render_context.py` verb renames + dropped flags) + a caller/test-update list. **Application batched into the final cross-doc consistency pass** (matching how 09 batched its carry-backs).

- **`migration-checklist.md`** — public-symbol ledger is complete. Final implementation rows are
  all `[x]` or `[-]`; no `[ ]`, `[/]`, or `[?]` rows remain after the final audit.

- **`open-questions.md`** — closed design-decision record. All design-era open items were
  resolved, applied, deferred, or dropped before implementation planning.

### Closed sub-spec sequence

**Decomposition (2026-05-25):** during sub-spec 09 the old `experiment/` mega-domain was
**split into three peer domains** — `experiment/` (Experiment noun + cross-trial views),
`trial/` (Trial noun, promoted to a top-level peer), `agent/` (the agentic loop). So
sub-spec 09 became **three** sub-specs (09 / 10 / 11). The split was carried back into `00`
(§5/§6/§7/§8.6–§8.8/§11.1/§12/§13.1/§16/§17, Appendix A) and `02` (Trial type homes +
`run_url`/`artifact_url` + diagnostics zero-file). See `00` §8.6 decomposition note + §17.12.

| # | Topic | Why this position | Status |
|---|---|---|---|
| 02 | **MLflow seam interfaces** | Every domain calls into MLflow; lock function signatures + return types first. | **DONE** |
| 03 | **Cleanup orchestration** | Touches every persistence layer; cascade order is load-bearing. | **DONE** |
| 04 | **Validate framework + registry** | Decides eager-vs-lazy registry import; every domain has a `checks.py`. | **DONE** |
| 05 | **Data domain** | Two Tier 3 anchors (`DataSource` + `DataPipeline`); lots of moving parts. | **DONE** |
| 06 | **Model domain** | Smallest but most-important Tier 3 anchor (`BaseModel`); + project-mandated preprocessing contract. | **DONE** |
| 07 | **Eval domain** | `Metric` ABC + EvalDataset. | **DONE** |
| 08 | **Runner domain** | Straight-line trial chain. | **DONE** |
| 09 | **Experiment domain** (slimmed) | Experiment noun + overview state + cross-trial views (leaderboard/compare/summary). | **APPROVED 2026-05-25** (§18 carry-backs batched with 10 closeout) |
| 10 | **Trial domain** | Trial noun, promoted to a peer: create/fork/promote/cleanup/packaging/metadata, show_trial/load_model, TrialSummary/TrialDetails types. | **DONE 2026-05-26** |
| 11 | **Agent domain** | The agentic loop (relocate-only): launch (was run_loop), timeline (was hook), Proposal contract, proposer_context. | **DONE 2026-05-27** |

All sub-specs are done, the implementation plan exists in `../plan/`, and Phases 0-7 plus the
final whole-refactor audit are complete. Use this section as the historical design sequence, not
as live next-action guidance.

---

## Sub-spec workflow (historical pattern from sub-specs 01 + 02)

No sub-spec remains open. This workflow is retained as historical process context:

1. **Find and read the relevant legacy code.** Identify which folders/files in `automl_legacy/`
   (and possibly `hooks/`, skills material, or project fixtures) correspond to the sub-spec's
   topic. Read them. Understand what the legacy code does — its public surface, contracts, edge
   cases — so every recommendation is grounded in what already exists. **Most of the design is
   reshaping current behavior into the new structure, not inventing.**
2. **Open with the most upstream question.** The first question is usually about *vocabulary*, *scope*, or *contract shape* — whatever decision other decisions depend on. State current behavior, then propose options + recommendation.
3. **Walk down the design tree, one question at a time.** Each question: 2-3 lettered options, a clear recommendation, reasoning. Wait for the user's pick before going to the next. If the user pushes back on a recommendation or surfaces a new concern, drill into THAT before continuing.
4. **Apply edits incrementally** to the in-progress `NN-<topic>.md` as decisions settle. Don't batch to the end.
5. **Three-agent parallel review BEFORE locking** (load-bearing — caught real issues for sub-spec 02):
   - Agent 1: independent fresh-eyes review (`feature-dev:code-reviewer`) — knows only the goal, not the conversation
   - Agent 2: codebase comparison (`feature-dev:code-explorer`) — current behavior vs. proposed surface; gap detection
   - Agent 3: coverage validation (`general-purpose`) — was the conversation faithfully captured?
   - Launch in parallel, foreground. Report findings to user. Flag false positives explicitly. User decides what to fix.
6. **Apply review findings.** User may delegate decisions; honor their preferences (see below).
7. **Closeout, in this order:**
   - Update the sub-spec doc with final edits
   - Mark resolved items in `open-questions.md` (🟡 → 🔵 or ⚪ or ⚫)
   - Flip `[?]` rows to `[ ]` in `../plan/migration-checklist.md` for symbols now covered
   - Update this README's "What's done" section to add the new sub-spec; flip its row in the sequence table to **DONE**

## User preferences distilled from sub-specs 01 + 02 + 03

These came up repeatedly. Apply by default; ask only if a decision feels in tension with them.

- **`dry_run` is a container, not a per-operation parameter.** Top-level CLI flag (`automl --dry-run …`), maps to `Session.dry_run`. The two universes (real / dry_run) are strictly isolated — separate MLflow namespaces, separate GCS prefixes, separate local cache paths, separate trial sandbox dirs. **Any operation (cleanup, run, profile, …) cleans / writes / reads ONE universe per invocation.** No per-verb `--mode {real,dry_run}` flags. No `--both` shortcuts. No `dry_run: bool` library function parameters. To act on both universes, run the command twice. (Stated repeatedly through sub-spec 03; if a future session drifts back into "delete both modes" framing, refuse.)

- **Simple, native, no over-abstraction.** When in doubt, fewer concepts. (Sub-spec 02: `Route` and `ActiveTrial` class were both proposed, then deleted when the user pushed back.)
- **Don't overbuild — but DO carry forward what already works.** Most functionality exists in
  `automl_legacy/`; the refactor reshapes it into the new structure, it does NOT rewrite
  everything from scratch. When a legacy feature has a clear home in the new design, port it.
  When a feature is speculative or "future-proofing" code with no real consumer, leave it out.
  Per memory `feedback_extension_points_follow_demand`.
- **Schema strategy is locked: additive-only + `schema_version: int = 1` + `from_dict` loader that strips unknown keys.** Carry this exact pattern into every new typed schema.
- **Clean cut, no back-compat for persisted state.** Old tag values (e.g. `'success'`), old run formats, old paths — won't be readable by new code. Per memory `feedback_no_back_compat`.
- **Recommend ONE answer, not menus.** When a clear best exists, lead with it. Per memory `feedback_recommend_dont_punt`.
- **Each sub-spec is a focused interview.** Walk the design tree one question at a time. If you're heading toward many open branches, surface that — scope may be too broad and need splitting.

## Out of scope (intentionally not in this refactor)

If a new session hits any of these, the answer is "out of scope, leave in legacy or defer to a future sub-spec":

- **Project-level "learning" subsystem.** Today's `store.py` has `write_learning_cache`, `write_cache_json`, `learning_feature_payloads`, golden/weak feature artifacts. These remain in `automl_legacy/` and do NOT migrate. Per user direction — "not well thought out yet." See `02-mlflow-seam.md` §13 item 12.
- **Proposer-context composite** (`get_context` in legacy `store.py`). The 500-line aggregator gets rebuilt domain-side in `agent/proposer_context.py` (sub-spec 11 territory — post-decomposition), not in the mlflow seam.
- **runs-using-strategy, runs-in-metric-band** — no-caller analytical queries. **No placeholder file** (sub-spec 09 §Q4); add a seam search + view helper on real demand. (NOTE: `recent_failures` and `compare` are **in scope** — 00 §11.1 ships `compare`, and the proposer-context consumes `recent_failures`; they live in `experiment/views/`, not deferred.)
- **Backward-compat shims, migration helpers, dual-write paths.** None. Clean cut.
- **Distributed tracing, observability, retry/backoff for transient failures.** Out of scope unless a sub-spec explicitly requests them.

---

## Resuming → see the front door

Design is frozen and the build closeout is tracked in [`../plan/`](../plan/). **To resume the
refactor, start at [`../README.md`](../README.md)** (status + next action + doc map) — it
supersedes the design-era "continue with the next sub-spec" flow this section used to describe.
(The behavior-level companion this README once flagged as a follow-on now exists:
[`../plan/acceptance-checklist.md`](../plan/acceptance-checklist.md).)

---

## The interview style we've been using

The user prefers a particular style for design conversations on this project:

> "Interview me relentlessly about every aspect of this plan until we reach a shared understanding. Walk down each branch of the design tree, resolving dependencies between decisions one-by-one. For each question, provide your recommended answer. Ask the questions one at a time. If a question can be answered by exploring the codebase, explore the codebase instead."

In practice this means:

- One focused question per message
- Each question has 2–3 lettered options
- Each question includes a clear recommendation with reasoning
- Walk from upstream decisions (e.g., vocabulary) to downstream ones (e.g., file names)
- When the user pushes back or surfaces a new concern, drill into THAT before continuing the tree
- Explore the codebase rather than ask when the answer is in the code

The user also prefers:
- Concise responses (per `feedback_recommend_dont_punt` memory — recommend ONE answer, not menus of 4+)
- No back-compat hacks (per `feedback_no_back_compat`)
- Project-local files mirror core structure (per `feedback_project_mirrors_core`)
- Extension points follow real demand (per `feedback_extension_points_follow_demand`)
- Tier 3 abstractions only where there's a real subclassing need

---

## Workspace context (one-line orientation)

- `automl_dev-refactor/` — active refactor worktree on branch `refactor/four-layer`; contains the cutover `automl/` package.
- `automl_dev/` — original working tree; source of the copied `.env` for local MLflow auth.
- `kaggle_home_credit/` — test harness sandbox at `/Users/zhengisamazing/1.python_dir/brigit/`
- `automl_runs/` — per-test-iteration working copies
- `mlflow_local/` — local MLflow tracking server (`mlflow_local start` to launch; URI `http://127.0.0.1:54321`)
- See `/Users/zhengisamazing/1.python_dir/brigit/CLAUDE.md` for full workspace conventions.

---

## File index

```
docs/superpowers/automl-refactor/
├── README.md                       ← ★ FRONT DOOR (status, doc map, what-next)
├── spec/                           ← design (this folder)
│   ├── README.md                   ← this file — design index
│   ├── 00-structural-design.md     ← parent spec (vocabulary, layers, folder shape)
│   ├── 01..11-*.md                 ← the 12 sub-specs (all approved)
│   └── open-questions.md           ← CLOSED design-decision record
└── plan/                           ← execution (how / order / status / next)
    ├── implementation-strategy.md  ← the overarching plan (approach, dep graph, phases 0–7)
    ├── migration-checklist.md      ← symbol-coverage ledger (living)
    ├── acceptance-checklist.md     ← behavior gates (living)
    └── phases/                     ← per-phase detailed plans (just-in-time)
```

All sub-spec files (00–11) are approved and the design is frozen. The `pending/` superseded
drafts were deleted (2026-05-27). **For build status + the next action, see
[`../README.md`](../README.md) and [`../plan/`](../plan/).**
