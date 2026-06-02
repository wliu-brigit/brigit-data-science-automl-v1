# brigit-automl — Structural Refactor Design

**Date:** 2026-05-19
**Status:** Design approved; implementation is tracked in `../plan/` (Phase 1 complete).
**Scope:** Restructure the `brigit-automl` Python package so the conceptual model is consistent, language no longer drifts, and code has a single unambiguous home.

This spec is a point-in-time design artifact. Treat the code in the refactor worktree's fresh
`automl/` package as the source of truth for current behavior once implemented; treat this spec
as the rationale and shape it was built to.

---

## 1. Context

`brigit-automl` is a Claude Code plugin that runs an agent-driven AutoML loop. The library hands trials to a proposer agent (decides what to try next from MLflow context) and a coder agent (writes `model.py` for one trial and runs it). State lives in MLflow; heavy bytes live in GCS.

After several refactors the package has reached ~19k lines across 16 top-level folders. Vocabulary has drifted (the word *snapshot* names three different concepts; *run* names four; `loader.py` vs `loading.py` differ by one letter but do entirely different work). Some folders are sprawling (`runner/_execute.py` 1496L, `data/pipeline.py` 1262L, `cleanup.py` 1087L); some are dormant (`utils/`). Adding a new check or extending a data source means hunting across multiple folders to find the right hook. The test suite has grown to ~992 tests across ~140 files.

The user has stated the goal explicitly: *"I know where to find it with confidence."* That requires a stable conceptual model — not just nicer folder names.

## 2. Goals

1. **One conceptual model.** Six canonical nouns (Project / Dataset / Experiment / Trial / Proposal / Model) at one level; four architectural layers (Surface / Domain / Framework / Utility) at another.
2. **One home per piece of code.** Every file has an unambiguous domain; a decision rule decides where new code lands.
3. **Three API tiers inside the library.** Tier 1 facade (`automl.<thing>`), Tier 2 domain submodules (`automl.data.<thing>`), Tier 3 contracts (base classes / ABCs).
4. **One surface stack above the library.** Skill → CLI → library; never skill → library directly.
5. **The MLflow seam is explicit.** Domain code never `import mlflow` directly — it calls `automl.mlflow.<noun>` which returns typed domain objects.

## 3. Non-goals

- Changing the agent-driven loop behavior. Proposer/coder remains the iteration model; MLflow remains the durable state contract.
- Changing the project recipe contract (`TASK / DATA / EVAL / RUN_CONFIG` in `projects/<name>/config.py`).
- Adding new features.
- Replacing cloudpickle as the model serialization format.
- Replacing MLflow as the persistence backend (the seam is for clarity, not pluggability).
- Snowflake remains stubbed in the dev workspace; the override hook in `DataPipeline.load_training_data` remains the sanctioned harness escape.

## 4. Strategy: side-by-side rebuild

The pre-refactor `automl/` package is frozen as `automl_legacy/` in the refactor worktree. An
in-place refactor of 19k lines and ~992 tests would require every intermediate state to keep the
harness green — a slow, error-prone path.

**Plan:**

1. Rename the legacy package `automl/` → `automl_legacy/`. Treat it as read-only reference.
2. Grow a fresh `automl/` package on branch `refactor/four-layer` in worktree
   `automl_dev-refactor/`. New package, new tests.
3. Keep the legacy tree only as a reference while the new one reaches parity on the Home Credit
   harness; new code never imports from `automl_legacy/`.
4. Cut over: `automl_legacy/` is deleted, leaving exactly one active package.

The branch/worktree ensures no work is lost while the fresh package is built.

## 5. Canonical vocabulary

The six nouns. Each is a class in the library and a level in the system's state hierarchy.

| Noun | Definition | Identity | Lifecycle |
|---|---|---|---|
**Nouns vs. folders — they are not 1:1.** The six nouns are the conceptual vocabulary; the folder layout (§7) is organized by *responsibility*, not by noun. Five of the six nouns map to a domain folder of their own — Project→`project/`, Dataset→`data/`, Model→`model/`, Experiment→`experiment/`, and **Trial→`trial/`** (a top-level peer of `experiment/`, not a sub-folder of it — this reverses an earlier call; see §17.11 for why). The sixth noun, **Proposal**, has no folder of its own: it is the proposer↔coder contract and lives inside `agent/`. Two domain folders are not nouns at all — `runner/` (executes one trial) and `agent/` (the agentic loop). And `eval/` is a domain whose concept is a *functional component*, not one of the six lifecycle nouns. So: eight domain folders, six nouns, and they overlap but don't coincide. See §7 for the full folder shape.

| Noun | Definition | Identity | Lifecycle |
|---|---|---|---|
| **Project** | Persistent recipe + workspace for one prediction problem. | Folder name under `projects/`. | Created via scaffold; durable. |
| **Dataset** | Immutable training data view materialized as parquet in GCS. The parquet is the canonical form. | A composite `identity_hash` over: `data_content` + `feature_registry` + `schema` + `source_identity` component hashes (the typed `Dataset` + `ComponentHashes` in `data/dataset.py`, per sub-spec 05 Q4). | Created on demand by `data.materialize`; shared across experiments by hash. **Source *lineage*** (raw SQL query text, Snowflake table name, CSV/parquet paths) is logged alongside as traceability metadata; it does NOT participate in identity. **Source *identity*** (a stable hash of source config — bucket+prefix, table+as-of, etc.) does participate, so re-fetching from the same source produces the same Dataset. |
| **Experiment** | Long-lived, DS-named optimization container. Holds many trials of any strategy. | DS-chosen `experiment_id` string. | DS declares the name in `RUN_CONFIG.experiment_id`; explicit `automl experiment create <name>` then creates the MLflow experiment. Both required — a run fails if either is missing. Archived when done. |
| **Trial** | One model attempt within an Experiment. Absorbs the MLflow "run" concept — trial = MLflow run. | Numeric within experiment + slug. | Created with a Proposal, executed by the runner, possibly promoted. |
| **Proposal** | JSON contract from proposer agent to coder agent. Describes what to try next. | Persisted as a trial artifact. | Generated → validated → persisted → consumed by coder. |
| **Model** | Fitted artifact produced by a Trial. Subclass of `BaseModel`, serialized via cloudpickle. | Trial-scoped (one model per successful trial). | Fit → packaged → potentially promoted. |

Drift to fix:

- "Snapshot" formerly named three concepts (data/snapshot.py, eval/snapshot.py, profile/snapshot.py). The user-facing noun is now **Dataset** everywhere, and per sub-spec 05 Q1 "snapshot" is fully retired at the code level in the data domain (class names, field names, file names, GCS paths, MLflow tag keys). The eval-layer file becomes `eval/eval_dataset.py`; the profile logic collapses into a single `data/profile.py` (sub-spec 05 Q5 — not a `profile/` subfolder; MLflow writing moves to `mlflow/project/artifacts.py`).
- "Run" no longer exists as a separate noun. A "trial run" and an "MLflow run" are the same thing.
- "Experiment" is one concept (a long-lived working container). It is *not* a single hypothesis or strategy — it can hold trials of any strategy. Same word, no overload.

### 5.1 How the noun hierarchy maps to MLflow

MLflow's data model has two levels: *MLflow experiment* → *MLflow run*. Our three-level noun hierarchy (Project → Experiment → Trial) maps onto MLflow as follows:

| AutoML noun | MLflow representation | Naming |
|---|---|---|
| Project | A dedicated MLflow *experiment* for project-scoped state | `<project_name>/overview` |
| (project overview) | A *run* inside the project's MLflow experiment, holding project-wide artifacts: the Dataset index + manifests, **data profiles (EDA)** — moved here from experiment-overview per sub-spec 05 Q5 — and data-learning artifacts | run name: `overview` |
| Experiment | A dedicated MLflow *experiment* per AutoML Experiment | `<project_name>/<experiment_id>` |
| (experiment overview) | A *run* inside the experiment's MLflow experiment, holding session reconciliation reports + active-dataset pin | run name: `overview` |
| Trial | A *run* inside the experiment's MLflow experiment | trial slug + sequence number |

The project-level overview is where shared state spans Experiments — most importantly, the set of Datasets the project has materialized, plus EDA / data-learning artifacts that are about the project's data, not about any one experiment.

This mapping is implemented by `mlflow/_routing.py` (path construction) and the per-noun folders (`mlflow/project/`, `mlflow/experiment/`, `mlflow/trial/`) — see sub-spec 02 §4 for the authoritative seam layout.

## 6. Four-layer architecture

```
   ┌────────────────────────────────────────────────────────┐
   │   SURFACE STACK   skill  →  CLI  →  library facade     │
   └────────────────────────────────────────────────────────┘
                              ▼
   ┌────────────────────────────────────────────────────────┐
   │   DOMAIN LAYER  — eight domains, one per concept        │
   │      project · data · model · eval · runner ·           │
   │      experiment · trial · agent                         │
   └────────────────────────────────────────────────────────┘
                              ▼
   ┌────────────────────────────────────────────────────────┐
   │   FRAMEWORK LAYER  — cross-cutting, not nouns          │
   │      mlflow · validate                                  │
   └────────────────────────────────────────────────────────┘
                              ▼
   ┌────────────────────────────────────────────────────────┐
   │   UTILITY LAYER  — no AutoML opinion                    │
   │      utils  (io · logging · paths · errors)             │
   └────────────────────────────────────────────────────────┘
```

| Layer | What's in it | Tier-3 extension |
|---|---|---|
| **Surface** | `skills/`, `cli/`, top-level `automl/__init__.py` facade | none |
| **Domain** | The six noun owners; substantive logic + ABCs | yes — `BaseModel`, `DataSource`, `Metric` |
| **Framework** | Cross-cutting machinery every domain participates in | yes — Validate registry |
| **Utility** | Non-AutoML primitives — would make sense in any Python project | none |

## 7. Folder shape

```
automl/
├── __init__.py                  ← Tier 1 facade (small)
├── errors.py                    ← exception hierarchy (top-level for visibility)
│
├── project/                     ← DOMAIN: Project
│   ├── __init__.py              ← Tier 2 exports: ProjectConfig, Session, use_project, …
│   ├── config.py                ← ProjectConfig (loaded, validated, immutable) — see project-context sub-spec
│   ├── session.py               ← Session (active state, contextvar-held) — see project-context sub-spec
│   ├── scaffold.py              ← create projects/<name>/
│   ├── metadata.py              ← candidate resolution + diagnostics
│   ├── task.py                  ← Task types
│   ├── run_config.py            ← RunConfig, Splits, ModelsConfig, ModelRoute  (Splits replaces Split — sub-spec 05 Q8)
│   ├── dependencies.py          ← parse project pyproject.toml
│   ├── cleanup.py               ← cascading cleanup at project / experiment / trial scopes
│   ├── checks.py                ← project + config + env checks (registered)
│   ├── _import.py               ← private: config.py module import + sys.path juggling
│   └── _env.py                  ← env-var loading internals
│
├── data/                        ← DOMAIN: Dataset   (detailed in sub-spec 05)
│   ├── __init__.py              ← Tier 2 verbs + types
│   ├── dataset.py               ← Dataset, LoadedDataset, LoadedSlice, DatasetIndex, ComponentHashes
│   ├── spec.py                  ← DataSpec
│   ├── pipeline.py              ← DataPipeline  ← Tier 3 anchor (orchestration-side)
│   ├── registry.py              ← list_datasets / load_dataset / load_dataset_by_id / load_dataset_by_trial
│   ├── split.py                 ← hashing MECHANISM: HashKey, hash_key_columns, add_split_id, split_report
│   │                              (Splits CONFIG type lives in project/run_config.py — sub-spec 05 Q8)
│   ├── contract.py              ← TrialDataContract, TrialRef, DatasetRef, SliceContract + validators
│   ├── features.py              ← FeatureRegistry, FeatureEntry (moved from core/)
│   ├── profile.py               ← deterministic profiler (single file; pluggable check list) + Profile type
│   ├── checks.py                ← data + contract checks (registered)
│   └── sources/                 ← DataSource ABC + builtins
│       ├── __init__.py
│       ├── base.py              ← DataSource ABC  ← Tier 3 anchor (source-side)
│       ├── snowflake.py
│       ├── local_csv.py
│       └── gcs_parquet.py
│   (no data/adapters/ — legacy *Pipeline wrappers deleted per sub-spec 05 Q2)
│
├── model/                       ← DOMAIN: Model
│   ├── __init__.py
│   ├── base.py                  ← BaseModel ABC  ← Tier 3 anchor
│   ├── preprocessing.py         ← RequiredTransformer + SklearnTransformer protocol + describe_required_transformers (sub-spec 06 — project-mandated preprocessing contract)
│   ├── packaging.py             ← save_model (cloudpickle write only; load_model → trial/show.py)
│   └── checks.py                ← model probe / pre-fit smoke + check_required_transformers (the §04-deferred "model" check, re-added by 06)
│
├── eval/                        ← DOMAIN: Eval
│   ├── __init__.py
│   ├── base.py                  ← Metric ABC + EvalSpec  ← Tier 3 anchor
│   ├── metrics.py               ← Auc, LogLoss, ThresholdSweep (builtins)
│   ├── evaluate.py              ← evaluate() verb
│   ├── eval_dataset.py          ← EvalDataset identity (was eval/snapshot.py)
│   ├── prepare.py               ← prepare eval dataset / augmentation / split view
│   ├── results.py               ← EvalResult / EvalIndex / Predictions schemas
│   ├── checks.py                ← eval + metric checks (registered)
│   └── _load.py                 ← (was eval/loader.py + eval/loading.py merged)
│
├── runner/                      ← DOMAIN: Runner
│   ├── __init__.py              ← exports: TrialResult, run_trial
│   ├── trial.py                 ← THE chain (straight-line; data → fit → eval → log)
│   ├── session_lock.py
│   └── template.py              ← model.py template text
                                   (no stages/ folder today — introduced only when
                                   a second runner type appears and primitives
                                   actually need sharing; see §17.2)
│
├── experiment/                  ← DOMAIN: Experiment   (the noun; cross-trial reads + overview state)
│   ├── __init__.py
│   ├── lifecycle.py             ← create experiment (archive deferred — no caller)
│   ├── store.py                 ← ExperimentOverview state (talks to mlflow/experiment)
│   ├── cleanup.py               ← experiment-scope cleanup; thin wrapper → project.cleanup
│   ├── checks.py                ← experiment-state checks (registered)
│   └── views/                   ← read paths over experiment history (cross-trial)
│       ├── __init__.py
│       ├── types.py             ← LeaderboardData, ComparisonResult, MetricDelta
│       ├── leaderboard.py
│       ├── compare.py           ← composes trial.show_trial per run
│       ├── summary.py           ← build_summary + experiments-listing enrichment
│       └── queries.py           ← recent_failures, strategies_attempted (compose seam)
│
├── trial/                       ← DOMAIN: Trial   (one model attempt; lifecycle + per-trial reads)
│   ├── __init__.py
│   ├── create.py                ← build the trial folder (path helper from runner; seed via seam)
│   ├── fork.py
│   ├── promote.py               ← human model → runner.run_trial
│   ├── cleanup.py               ← trial-scope cleanup; thin wrapper → project.cleanup
│   ├── packaging.py             ← package_model (notebook class → model.py source; stdlib-only)
│   ├── metadata.py              ← TrialMetadata, SeedSelection, ModelSource, TimingReport, ManifestEntry, TrialManifest schemas
│   ├── show.py                  ← show_trial, load_model
│   └── types.py                 ← TrialSummary, TrialDetails, TrialStatus, ParentExperimentRef (seam returns these)
│                                   (no checks.py — zero trial-state checks + no `validate trial` target; sub-spec 10 Q9)
│
├── agent/                       ← DOMAIN: the agentic loop   (relocate-only; see §8.8 + §17.12)
│   ├── __init__.py
│   ├── launch.py                ← build the agent subprocess invocation (was cli/run_loop.py;
│   │                              model routing, --agents JSON, env injection)
│   ├── timeline.py              ← reconcile agent hook events into MLflow (was hooks/agent_timeline.py;
│   │                              hooks/ becomes a thin stub)
│   ├── proposal.py              ← Proposal contract (proposer↔coder handoff) + field constants
│   ├── proposer_context.py      ← agent-facing context assembly (the proposer's input packet)
│   └── checks.py                ← proposal_schema (registered)
│
├── mlflow/                      ← FRAMEWORK: persistence layer
│   ├── __init__.py
│   ├── client.py                ← MLflow client + tracking URI
│   ├── _routing.py              ← <project>/<experiment_id> path rules + dry_run/ prefix + agent-events GCS prefix (private; matches sub-spec 02 §4)
│   ├── tags.py                  ← canonical tag-key constants
│   ├── code_bundle.py
│   ├── project.py               ← project-overview run read/write
│   ├── experiment.py            ← experiment-overview run + experiment queries
│   ├── trial.py                 ← start_trial / log_* / fetch trial details
│   ├── queries.py               ← shared search helpers
│   └── artifacts/               ← per-artifact persistence
│       ├── __init__.py
│       ├── start.py
│       ├── data.py
│       ├── eval.py
│       ├── failure.py
│       ├── features.py
│       ├── manifest.py
│       ├── model.py
│       ├── predictions.py
│       ├── timing.py
│       └── validation.py
│
├── validate/                    ← FRAMEWORK: validation orchestration + dispatch
│   ├── __init__.py
│   ├── base.py                  ← Issue, ValidationReport, Severity   (no CheckSpec — deleted, sub-spec 04 Q1)
│   ├── targets.py               ← orchestrators (project / model / proposal only — sub-spec 04 Q3); built-in checks called directly (no @register); also _import_project_validators + _RESET_FOR_TESTS
│   └── synthetic.py             ← test-fixture helpers
│                                   (no registry.py — deleted per sub-spec 04 Q1; check code lives in <domain>/checks.py, imported + called directly)
│
├── utils/                       ← UTILITY: non-AutoML primitives
│   ├── __init__.py
│   ├── logging.py
│   ├── paths.py
│   ├── hashing.py              ← content / schema / json hash primitives (shared by data + eval)
│   └── io/
│       ├── __init__.py
│       ├── gcs.py
│       └── snowflake.py
│
└── cli/                         ← SURFACE: thin verb wrappers
    ├── __init__.py              ← register + dispatcher
    ├── __main__.py
    └── <verb>.py                ← one file per user-intent verb
                                   (exact verb list deferred to implementation plan)
```

**Folder counts:** 8 domain folders + 2 framework folders + 1 utility folder + 1 surface folder + `errors.py` + `__init__.py` = 13 top-level entries, down from 16.

## 8. Per-domain specification

For each domain: what it owns, what it exposes at Tier 2, what its Tier 3 anchor is (if any), what it imports from outside its folder.

### 8.1 `project/`

- **Owns:** ambient project context (active project, repo root, config-module import), project scaffolding (create new `projects/<name>/`), config types (`Task`, `RunConfig`, `Splits`, `ModelsConfig`), env-var loading, **project-local pyproject.toml dependency parsing** (in `dependencies.py`), project + config + env validation checks, **cascading cleanup at all three scopes (project / experiment / trial)**.
- **Tier 2 exports:** `ProjectConfig`, `Session`, `use_project`, `session`, `active_session`, `clear_session`, `RunConfig`, `Splits`, `Task` types, `cleanup`. (`Splits` replaces today's `Split` — free-form named ranges per sub-spec 05 Q8; lives here, not in data/, to keep `data → project` dependency direction.) (Detailed contracts in the project-context sub-spec.)
- **Tier 3 anchor:** none (project config / session are data + active state, not extension).
- **Outbound deps:** `errors`, `utils`, `mlflow/` (cleanup orchestrates MLflow + GCS + local-file deletes).
- **Inbound:** every other domain (project Session threads through everything via contextvar; cleanup is invoked by CLI `automl cleanup`).
- **Why cleanup lives here:** the cleanup cascade starts from Project — deleting a Project removes everything under it; deleting an Experiment removes the trials under it. Project is the conceptual root of the cascade. `experiment/cleanup.py` and `trial/cleanup.py` are thin wrappers that call `project.cleanup(scope=..., ...)` to keep verb-namespace symmetry without duplicating logic.

### 8.2 `data/`

- **Owns:** Dataset identity + materialization, data sources (one file per source type under `sources/`), pipeline (the waterfall from source to materialized parquet), per-row split-id hashing mechanism, feature registry, deterministic profiler, trial data contract. (Split *config* — which named ranges are train/test — lives in `project/run_config.py`; data owns the *mechanism* that applies them.)
- **Tier 2 exports:** `Dataset`, `LoadedDataset`, `LoadedSlice`, `DatasetIndex`, `DataSpec`, `DataSource`, `DataPipeline`, `FeatureRegistry`, `FeatureEntry`, `Profile`, `TrialDataContract` (+ `TrialRef`, `DatasetRef`, `SliceContract`), and verbs `build_dataset`, `materialize`, `list_datasets`, `load_dataset`, `load_dataset_by_id`, `load_dataset_by_trial`, `profile`, `get_profile`. (Detailed contracts in sub-spec 05.)
- **Tier 3 anchors (two):** `DataSource` (`data/sources/base.py`) for source-side variation; `DataPipeline` (`data/pipeline.py`) for orchestration-side variation. Composition assembles them via `DataSpec(source=…, pipeline_cls=…)` — sub-spec 05 Q2.
- **Outbound deps:** `project` (Session, RunConfig, Splits), `mlflow.project.artifacts`, `mlflow.trial.artifacts`, `utils.io`, `utils.hashing`.
- **Inbound:** `runner`, `eval`, `experiment`.

### 8.3 `model/`

- **Owns:** the model contract (`BaseModel` ABC), serialization (`model/packaging.py::save_model` — cloudpickle write only), pre-fit probe checks, and the **project-mandated preprocessing contract** (`RequiredTransformer` + `SklearnTransformer` protocol + `describe_required_transformers` in `model/preprocessing.py`; the framework gate `check_required_transformers` in `model/checks.py`) — sub-spec 06. **Not owned (06 correction — model deps = `errors` only):** `load_model` (MLflow PyFunc load, needs the seam) → `trial/show.py`; `package_model` (notebook class → `model.py` source extraction) → `trial/packaging.py`.
- **Tier 2 exports:** `Model`, `BaseModel`, `RequiredTransformer`, `describe_required_transformers` (sub-spec 06).
- **Tier 3 anchor:** `BaseModel` (`model/base.py`).
- **Outbound deps:** `errors`.
- **Inbound:** `runner` (instantiates Model + calls `save_model` in trial chain).
- **Notes:** thin today, can expand as model lifecycle grows (e.g. serving manifest, signature inference).

### 8.4 `eval/`

- **Owns:** Metric ABC + builtins, EvalSpec, EvalDataset (the held-out evaluation view; distinct from training Dataset), the `evaluate` verb, eval results schema, predictions schema, compatibility checks.
- **Tier 2 exports:** `Metric`, `EvalSpec`, `EvalDataset`, `evaluate`, `evaluate_frame`, `EvalResult`, `Predictions`, `Auc`, `LogLoss`, `ThresholdSweep`. (Sub-spec 07: `EvalResult` is singular — the `EvaluateResult`/`EvalResults` pair was consolidated into one type.)
- **Tier 3 anchor:** `Metric` (`eval/base.py`).
- **Outbound deps:** `project`, `data` (concept-level + public slice/index APIs — see below), MLflow trial artifact writers/readers for eval and predictions, `utils.io`, `utils.hashing`.
- **What `eval` imports from `data` (legitimate, public-only):** `list_datasets` to resolve parent Dataset metadata while preparing a split-view EvalDataset; `load_dataset_by_id` (the public slice loader — a `split_view` EvalDataset delegates bucket realization to `data.load_dataset_by_id(of_dataset_id, split_range=buckets)` per sub-spec 07 Q1; integrity from the content-addressed id + data's L2 load-time validation). Content-hash primitives (`dataframe_content_hash`, `schema_hash`, `json_hash`) come from `utils/hashing.py` (public), NOT from `data/` — see §13.8. **No underscore-prefixed privates cross this boundary.** This is eval's only *runtime data-loading* dependency on data — sub-spec 07's tripwire: if it thickens, re-evaluate the §13.8 Dataset/EvalDataset unification. (Sub-spec 07 settled eval-side naming: `of_data_snapshot_id` → `of_dataset_id`; the data domain no longer exposes "snapshot"-named symbols.)
- **Inbound:** `runner` (eval stage), `experiment.views` (re-eval against new EvalDataset).

### 8.5 `runner/`

- **Owns:** the deterministic chain that executes one trial (`data → fit → eval → log`) as a straight-line procedure in `runner/trial.py`, the model.py template, the session lock.
- **Tier 2 exports:** `TrialResult`, `run_trial`.
- **Tier 3 anchor:** none. There is no stage abstraction today. `runner/trial.py` is a straight-line file (~600–1000 lines after slimming from today's `_execute.py` + `_stages.py`).
- **Outbound deps:** `project`, `data`, `model`, `eval`, `validate`, `mlflow.trial`.
- **Inbound:** `trial.create` / `trial.promote` (invoke runner), CLI `automl trial run`.
- **Future runner types:** when a second chain (HPO, ablation, etc.) actually needs to exist, refactor *at that time*: introduce `runner/_stages.py` for the shared primitives that the two chains both use, and add `runner/<new>.py` as a sibling. **Don't pre-build the stage abstraction.** Per `feedback_extension_points_follow_demand` — extension shapes follow real demand.

> **Decomposition note.** The legacy `experiment/` mega-domain — which had absorbed Trial operations, the Proposal contract, every read view, the agent-loop launcher, and the agent-timeline reconciliation — is dissolved into **three peer domains**: `experiment/` (the Experiment noun + cross-trial reads), `trial/` (the Trial noun), and `agent/` (the agentic loop). MLflow itself treats Experiment and Run as separate entities, and the project→experiment→trial *hierarchy* is resolved by the seam (`mlflow/{project,experiment,trial}` routing + tags) and the runner at execution time — not by folder nesting. The agent loop is extracted because it is the core engine, and a clean boundary is what lets a different agent/driver be wired in later (deferred — §17.12). See §17.11 for the reversal of the earlier "Trial is a sub-folder" call. Sub-specs 09 / 10 / 11 settle these three.

### 8.6 `experiment/`

- **Owns:** experiment lifecycle (create; archive deferred — no caller), experiment-overview run state (`ExperimentOverview`), experiment-scope cleanup wrapper, and the **cross-trial read views** (leaderboard, compare, summary, experiments-listing, and the `recent_failures` / `strategies_attempted` aggregations).
- **Doesn't own:** Trial operations (→ `trial/`); the Proposal contract + agent loop (→ `agent/`); cascading cleanup (→ `project/cleanup.py`). `experiment/cleanup.py` is a thin wrapper that translates the experiment id into a call against the shared cascade engine (sub-spec 03).
- **Internal structure:** top-level for lifecycle + store + cleanup + checks; `views/` subfolder for cross-trial read paths.
- **Tier 2 exports:** `Experiment` (alias of `ExperimentOverview`), `create`, `leaderboard`, `compare`, `build_summary`.
- **Tier 3 anchor:** none.
- **Outbound deps:** `project` (Session / cleanup-wrapper delegation), `trial` (`compare` / `summary` compose `trial.show_trial`; views return `trial.TrialSummary`/`TrialDetails`), `data` (active Dataset pin via seam), `mlflow.experiment`, `mlflow.trial`.
- **Inbound:** CLI verbs (`experiment leaderboard` / `compare` / `summary` / `delete`), `agent.proposer_context` (composes the views).

### 8.7 `trial/`

- **Owns:** the Trial noun — trial lifecycle (create / fork / promote), trial-scope cleanup wrapper, `package_model` (notebook class → `model.py` source extraction; stdlib-only — the 06 correction, *not* model serialization), trial metadata + seed schemas, per-trial read paths (`show_trial`, `load_model`), and the seam-returned trial types.
- **Doesn't own:** trial *execution* — that is `runner/` (the straight-line chain). `trial.create` builds the folder (using a path helper from `runner/`) and `trial.promote` calls `runner.run_trial`; the runner runs it. Cascading cleanup lives in `project/cleanup.py`; `trial/cleanup.py` is a thin wrapper.
- **Internal structure:** top-level files per operation + `metadata.py` (write-side schemas) + `show.py` (reads) + `types.py` (seam-returned read contracts). **No `checks.py`** (sub-spec 10 Q9 — zero trial-state checks, no `validate trial` target; add on demand).
- **Tier 2 exports:** `create`, `fork`, `promote`, `delete` (cleanup), `package_model`, `show_trial`, `load_model`, `TrialSummary`, `TrialDetails`.
- **Tier 3 anchor:** none.
- **Outbound deps:** `project` (Session / cleanup-wrapper delegation), `runner` (path helper + `run_trial`), `eval` (`EvalResult` type for `TrialDetails.evaluations` — sub-spec 10 Q4; acyclic, eval does not import trial), `utils` (`SLUG_RE` — sub-spec 10), `mlflow.trial`, `mlflow.experiment` (seed selection + `next_trial_number` via seam — never `import mlflow`).
- **Inbound:** CLI verbs (`trial show` / `delete` / `run`), `experiment.views` (imports trial types; `compare`/`summary` call `show_trial`), `agent` (`create` consumes a validated proposal; `proposer_context` reads trial summaries via seam), `runner` (imports trial *types* — the type-vs-function split keeps `trial ↔ runner` acyclic, per sub-spec 08).

### 8.8 `agent/`

- **Owns:** the agentic loop — the agent **launcher** (build the one `claude` subprocess invocation: model routing, `--agents` JSON, env injection; was `cli/run_loop.py`), the agent **timeline** reconciliation (consume Claude Code hook events → MLflow agent-run artifacts; was `hooks/agent_timeline.py`, which becomes a thin stub), the **Proposal** contract (proposer↔coder handoff) + its `proposal_schema` check, and the **proposer-context** assembly (the input packet the proposer reads).
- **Doesn't own:** the *sequencing* of proposer→coder — that stays **LLM-driven** (one subprocess drives the loop; `agent/` defines, launches, and observes agents, it is not a state machine). Trial creation/execution (→ `trial/` + `runner/`).
- **Scope for v1 = relocate, not redesign.** Move the launcher + timeline into the library, route their MLflow writes through the seam, apply the session convention. The internal timeline reconciliation algorithm is ported verbatim (00 §15.1). No driver abstraction, no agent registry, no multi-agent orchestration — those are deferred forward-looking work (§17.12).
- **Tier 2 exports:** `Proposal`, `build_launch`, `handle_event`, `publish`, `gather_proposer_context`. (`publish` added per sub-spec 11 #1 — `handle_event` alone undersold the timeline surface.)
- **Tier 3 anchor:** none today (the latent extension axes — more agents/roles, more drivers — are §17.12, built on demand).
- **Outbound deps:** `project` (Session / model routing from config), `experiment.views` + `trial` (`proposer_context` composes their reads), `data` (active Dataset / profile via seam), `runner` (`run_trial` is reached by `trial.promote`, not directly), `mlflow.project` / `mlflow.experiment` / `mlflow.trial`, `utils.io.gcs`, `validate` (`checks.py`).
- **Inbound:** CLI verbs (`experiment run` → `agent.build_launch`; `experiment proposer-context` → `agent.proposer_context`), `hooks/` stub (→ `agent.timeline.handle_event`).

## 9. Framework layer

### 9.1 `mlflow/` — the persistence seam

**Invariant: domain code never `import mlflow`.** It calls `automl.mlflow.<noun>.<verb>()`. The mlflow folder is the only place that knows MLflow's API.

Internal structure mirrors the noun hierarchy:

| File | Responsibility |
|---|---|
| `client.py` | MLflow client construction; tracking URI; env wiring |
| `_routing.py` | route string rules `[<namespace>/][dry_run/]<project>/<experiment_id>` — the optional `<namespace>/` segment (from `session.namespace`, the `--namespace` flag) and the conditional `dry_run/` segment (from `session.dry_run` — no `run_mode`/`"full_run"` string) are both path prefixes; deterministic agent-events GCS prefix from `(session, run_id)` (sub-spec 11 #6). Private — see sub-spec 02 §4. |
| `tags.py` | Canonical tag-key constants (one source of truth) |
| `code_bundle.py` | Stage code into MLflow at trial start |
| `project.py` | Project-overview run: read/write project-scoped tags + state |
| `experiment.py` | Experiment-overview run; experiment-level queries |
| `trial.py` | `start_trial`, `end_trial`, `log_*`, fetch trial details |
| `queries.py` | Shared `mlflow.search_runs` helpers |
| `artifacts/<thing>.py` | One file per artifact: validate against schema (imported from domain), persist to MLflow + GCS |

**Reader contract:** `mlflow/` readers return typed domain objects, not raw MLflow records. E.g. `mlflow.experiment.list_trials_with_metrics(ctx)` returns `list[TrialSummary]` (a type defined in `trial/types.py`). View code is thin formatting on top of typed objects.

**Dependency direction.** Domains call `automl.mlflow.<noun>` for IO; `mlflow/` imports types from each domain to return them as typed objects. This is a *one-way* dependency — no cycle (domains depend on mlflow's *functions*; mlflow depends on domains' *types*).

**Namespace note.** `automl/mlflow/` shadows the PyPI `mlflow` package within `automl.*`. Inside files under `automl/mlflow/`, `import mlflow` resolves to the PyPI package (absolute-import default); within-folder relative imports must use the explicit `.` syntax (`from . import client`). The invariant is precisely: **code outside `automl/mlflow/` never `import mlflow`** — code inside this folder is the only place the PyPI mlflow API is touched.

**Where do new types live? Strict rule: all *public* types returned from `automl.mlflow.<noun>` belong to domains.** No public MLflow-only types. `EvalResult` lives in `eval/`, `TrialSummary` in `trial/types.py`, `DataContract` in `data/` — and `mlflow/` returns these.

`mlflow/` may have **private internal types** (prefixed with `_` or kept in `mlflow/_internal.py`) for its own implementation — e.g. a `_RawRunRecord` it builds during a search before mapping into domain types. These are not part of the public API and must not appear in any function signature visible outside `mlflow/`.

So when in doubt: the type lives in the domain.

### 9.2 `validate/` — the validation framework

**Split:** framework code centralized; check functions live next to the types they check.

```
validate/               ← framework
  base.py               ← Issue, ValidationReport, Severity
  targets.py            ← project / model / proposal orchestrators (built-in checks called directly; also _import_project_validators + _RESET_FOR_TESTS)
  synthetic.py          ← test-fixture helpers

<domain>/checks.py      ← per-domain check functions (plain functions; orchestrators import + call them directly)
```

The DS still runs `automl validate project`; the verb dispatches to the target orchestrator, which **imports and calls each domain's `checks.py` functions directly** — no `@register` decorator, no `_CHECKS` global, no `registry.py` (all deleted in sub-spec 04 Q1). Each call is wrapped by a `_safe(name, fn, **kwargs)` helper so a crashing check emits an `Issue("error", "<name>.crashed", …)` instead of taking down the orchestrator. Per-project user checks are contributed via `projects/<name>/validators.py` exporting `PROJECT_CHECKS = {"project": [fn]}` (**`"project"` target only** — sub-spec 04 Q2).

## 10. Utility layer

```
utils/
├── logging.py          ← configure_logging (actually used)
├── paths.py            ← path helpers
├── hashing.py          ← content / schema / json hash primitives
└── io/
    ├── gcs.py          ← GCS client + read/write helpers
    └── snowflake.py    ← Snowflake session context manager
```

Test for membership: *Has zero AutoML semantics — would make sense in a random Python project.* If the answer is yes, it belongs here.

**`utils/hashing.py` specifically** holds primitives used by both `data/` and `eval/` to compute artifact identity: `dataframe_content_hash(df)`, `schema_hash(df)`, `json_hash(value)`. None of these encode any AutoML concept — they hash a DataFrame's content / a DataFrame's column-and-dtype schema / a JSON-serializable value, respectively. Any future content-identity primitive that's similarly AutoML-agnostic (e.g. a hash for a future calibration dataset or drift baseline) lands here as a sibling. See §13.8 for the design rationale.

`errors.py` stays at the package top (not under `utils/`) for visibility — the exception hierarchy is part of the package's public surface.

## 11. Surface layer

### 11.1 CLI

Verbs are organized by **user intent**, not by library shape. Each verb file in `cli/<verb>.py` is a thin argparse wrapper that calls 1–2 domain functions. **Zero business logic in CLI files.** The CLI is the same wrapper pattern repeated; logic always lives in the library it's wrapping.

#### Principles

1. **Thin wrappers.** Argparse + 1–2 library calls. If a verb's `.py` file grows past ~80 lines, that's a signal logic is leaking — refactor into the library.
2. **Intent-based grouping.** Sub-verbs cluster under a noun: `automl trial <action>`, `automl experiment <action>`, `automl validate <target>`. Avoid leaking library structure into verb names.
3. **Programmatic access via `--json`.** Every verb that produces structured output accepts `--json` and writes JSON to stdout. This is what skills/agents call — same library function, machine-readable framing.
4. **Lean — add verbs only when there's user demand.** No preemptive verbs for "things someone might want to do." A new verb arrives because a workflow needs it.
5. **No library `__main__` blocks.** Every operation reachable from the command line lands as a CLI verb; library files don't double as scripts.

#### CLI philosophy: noun-first, with one carve-out for `validate`

Every verb is `automl <noun> <action> [target]`, where `<noun>` is either a vertical-lifecycle noun (project, experiment, trial) or a functional-component noun (data, eval). The one cross-cutting carve-out is `automl validate <target>` — because `validate/` is a framework layer (§9.2), not a domain, and its CLI verb is the framework's entry point dispatching to per-domain checks.

This is the rule. Inconsistencies in the legacy CLI (`automl cleanup --scope X`, `automl run`, `automl inspect <thing>`, `automl proposal validate`, `automl session lock`, hyphenated `automl loop-context for-proposer`) collapse into noun-first form below.

#### CLI folder layout

`cli/<noun>.py` — **one file per noun**, with argparse subparsers for that noun's actions. Plus `cli/validate.py` for the cross-cutting verb. Legacy already follows this for `cli/trial.py` and `cli/project.py`; the rest of the noun files are net-new or renamed.

#### Top-level flags (session-wide environment modifiers)

A small set of flags sit on the `automl` entry point itself, **before any verb**. They modify the session that all verbs operate in — not the verb's behavior. The verbs themselves don't see these flags directly; they're picked up by the CLI dispatcher and threaded into `use_project(...)`.

| Flag | Type | Maps to | Effect |
|---|---|---|---|
| `--dry-run` | bool (presence) | `Session.dry_run` | All stateful verbs operate against the dry_run universe (different MLflow experiment namespace `dry_run/<project>/...`, different GCS prefix `gs://.../dry_run/<project>/...`, may use a data subset). Absent → real universe. |
| `--namespace <name>` | str (default `""`) | `Session.namespace` | Prefix every routed path with `<name>/` — a **full-universe** isolated sandbox (MLflow experiment names + GCS prefix + local trial sandbox dirs all get the `<name>/` segment). Orthogonal to `--dry-run` and composable (`<name>/dry_run/<project>/...`). Purpose: full-fidelity QA/test runs the user can clean up (`automl --namespace qa <noun> delete … --apply`) without touching the real (`""`) namespace. Renamed from legacy `route_namespace` (which was never wired). |

Rationale: dry_run is an *environment switch*, not a verb-specific argument. Every verb that touches state (cleanup, run, profile, materialize, etc.) routes the same way — just pointed at a different container. Putting the flag at the top level lets one flag apply uniformly to every verb without each verb redeclaring it.

```bash
# dry_run universe — all verbs route to dry_run namespace
automl --dry-run experiment delete X --apply
automl --dry-run experiment run
automl --dry-run data profile

# real universe (default, no flag)
automl experiment delete X --apply
automl experiment run
automl data profile
```

Future top-level flags belong here only if they're true session-wide environment modifiers (not verb-specific options).

#### Verb catalog (ships in v1)

Only verbs with a real caller today (skill-driven or required by the human dev workflow). Per principle 4 (lean — add verbs only when there's demand), preemptive verbs are explicitly deferred (next subsection).

| Noun | Verb | Library destination | Caller |
|---|---|---|---|
| **project** | `automl project list` | `project.metadata.list_projects` | human |
|  | `automl project init <name>` | `project.scaffold` | skill: setup |
|  | `automl project deps [--json]` | `project.dependencies` | skill: propose (replaces `python -m automl.core.dependencies`) |
|  | `automl project delete <name> [--apply] [--hard-delete] [--json]` | `project.cleanup` (project scope) | human |
| **experiment** | `automl experiment list` | `experiment.views.experiments` | human |
|  | `automl experiment run [<id>]` | `agent.build_launch` | skill: automl, setup (replaces top-level `automl run`) |
|  | `automl experiment delete <id> [--apply] [--hard-delete] [--json]` | `experiment.cleanup` (→ project.cleanup cascade) | human |
|  | `automl experiment leaderboard [<id>]` | `experiment.views.leaderboard` | skill: inspect (replaces `inspect leaderboard`) |
|  | `automl experiment compare <id1> <id2>` | `experiment.views.compare` | skill: inspect (replaces `inspect compare`) |
|  | `automl experiment summary [<id>]` | `experiment.views.summary` | skill: inspect (replaces `loop-context summary`) |
|  | `automl experiment proposer-context [<id>]` | `agent.proposer_context` | skill: propose (replaces `loop-context for-proposer`) |
| **trial** | `automl trial list [<experiment_id>]` | `mlflow.experiment.list_trials` | human |
|  | `automl trial run <dir>` | `runner.run_trial` | human; runner subprocess |
|  | `automl trial show <run_id>` | `trial.show` | skill: inspect (replaces `inspect show-trial`) |
|  | `automl trial delete <run_id> [--apply] [--hard-delete] [--json]` | `trial.cleanup` (→ project.cleanup cascade) | human |
|  | `automl trial lock {acquire,release}` | `runner.trial_lock` | skill: automl (replaces `python -m automl.session.lock`; renamed to fix `Session` naming clash from sub-spec 01) |
| **data** | `automl data list` | `data.registry.list_datasets` | human |
|  | `automl data profile [<dataset>]` | `data.profile` | skill: profile (replaces top-level `automl profile`) |
| **eval** | `automl eval list` | `eval.registry` (defined in sub-spec 07) | human |
|  | `automl eval compute --model-run-id … --eval-snapshot …` | `eval.evaluate` | human |
| **validate** *(cross-cutting)* | `automl validate <target>` | `validate.targets` (dispatch to per-domain `checks.py`) | skill: validate, setup, automl, propose |
|  | `target ∈ {project, model, proposal}` (sub-spec 04 Q3 froze three; `config`/`contracts`/`experiment` dropped). `automl validate proposal` also accepts `--output <path>` (writes the validated JSON on pass — sub-spec 04, keeps `render_context.py::persist_proposal` working). | | |

**Two `run` verbs are intentional.** `automl experiment run` launches the agent loop; `automl trial run <dir>` executes one trial subprocess. The noun disambiguates; both read naturally in context.

#### Verbs explicitly deferred (no caller today — add on real demand)

Per principle 4. These are conceptually fine but no skill or human workflow uses them yet, so building them now violates "follow demand."

- `automl project info` — currently no skill consumes structured project metadata via CLI; defer.
- `automl experiment create <id>` / `automl experiment archive <id>` — experiments are created by `experiment run` as part of bootstrap today; no separate creation verb is in demand. Archive likewise unused.
- `automl data materialize` — runner-driven via Python today; no skill/CLI use case yet.
- `automl trial create` / `trial fork` / `trial promote` — sketched in legacy CLI; no current skill/human caller. Defer until a workflow needs them.

When demand arrives, each adds as a sibling under its noun — no shape decision needed at that time.

#### Legacy → new mapping (CLI-only — for migration tracking)

| Legacy verb | New verb |
|---|---|
| `automl cleanup --scope project --project X` | `automl project delete X --apply` |
| `automl cleanup --scope experiment …` | `automl experiment delete <id> --apply` |
| `automl cleanup --scope trial --run-id Y` | `automl trial delete Y --apply` |
| `automl cleanup --apply --purge-mlflow` | `automl <noun> delete <id> --apply --hard-delete` |
| `automl run --project X` | `automl experiment run` (project from session) |
| `automl inspect leaderboard` | `automl experiment leaderboard` |
| `automl inspect show-trial` | `automl trial show` |
| `automl inspect compare` | `automl experiment compare` |
| `automl loop-context for-proposer` | `automl experiment proposer-context` |
| `automl loop-context summary` | `automl experiment summary` |
| `automl profile` | `automl data profile` |
| `automl project create` | `automl project init` |
| `automl proposal validate` | `automl validate proposal` (dedupe with the `validate` framework verb) |
| `python -m automl.session.lock <action>` | `automl trial lock <action>` |
| `python -m automl.core.dependencies` | `automl project deps` |
| `automl eval --model-run-id …` | `automl eval compute --model-run-id …` |

#### Programmatic access

Two access patterns over the same library functions:

- **CLI from a shell / skill:** `uv run automl <verb>` — for human and agent invocation.
- **Direct Python import:** `from automl.<domain> import <function>` — for notebooks, tests, library composition.

The library function is the canonical implementation. The CLI verb is a wrapper that adds argparse + output formatting. Anything you can do via CLI you can do via Python import, and vice versa — by design.

#### What this section explicitly defers to the implementation plan

- Exact argparse argument names (flag styles, default values).
- Whether some sub-verbs are bundled or split (e.g., `automl trial create-and-run` vs two verbs).
- Migration ordering — which verbs ship first, which `python -m <module>` invocations get cut over when.
- The skill prose changes that result from new verb names.

### 11.2 Skill → CLI seam

Skills shell out to CLI verbs. **Skills do not `import automl`.** This keeps the library's API stable as the agent's prose changes, and gives the loop one entry point per operation.

Skill-local `scripts/` (preflight, render_context) are the only place skills run Python directly; that Python imports from the library, never from another skill.

### 11.3 `hooks/` — Claude Code lifecycle integration

`hooks/` sits at the workspace root (peer to `automl/`, not inside it). It contains:

- `hooks.json` — Claude Code's wiring declaration (matchers + script targets).
- `hooks/agent_timeline.py` — a **thin stub** (~5–20 lines) that loads stdin and delegates to `automl.agent.timeline.handle_event(...)`. The actual reconciliation logic lives in the library at `agent/timeline.py` (see §8.8).

Why the split: today's `hooks/agent_timeline.py` is 1955 lines and imports legacy module paths by name; some of its functionality is even loaded by literal filesystem path. Moving the logic into `agent/timeline.py` makes it a normal library file (testable, importable, refactorable) and reduces the hook to a transport stub that only knows how to read Claude Code's stdin event format.

The full interface contract (event schema, MLflow artifact format, cache layout) is the **Priority 4 sub-spec** (§15.1).

### 11.4 `python -m <module>` policy: no library `__main__` blocks in the new design

The legacy codebase has 11 library files with `__main__` blocks (`automl/cleanup.py`, `automl/session/lock.py`, `automl/profile/snapshot.py`, `automl/mlflow/overview.py`, `automl/data/prepare.py`, `automl/core/dependencies.py`, etc.) — invoked as `python -m automl.session.lock acquire …` from skills. This pattern violates the thin-CLI invariant.

**New rule: zero library files have `__main__` blocks.** Every `python -m <module>` invocation becomes a `automl <verb>` CLI verb. Skills migrate to use `uv run automl <verb> …` exclusively. The exact verb names are in §11.1's catalog above (see the legacy→new mapping for `python -m automl.session.lock` → `automl trial lock` and `python -m automl.core.dependencies` → `automl project deps`).

## 12. Tier 1 facade — `automl.<thing>`

Sklearn-style minimal surface. The DS in a notebook does `automl.use_project("foo")` and then reaches into domains for everything else (`from automl.data import materialize`, `from automl.experiment import leaderboard`).

What `automl/__init__.py` exports (sketch — exact list deferred to implementation plan; project surface aligns with the project-context sub-spec):

```python
from automl.project import (
    ProjectConfig, Session,
    use_project, session, active_session, clear_session,
)
from automl.data import Dataset
from automl.experiment import Experiment
from automl.agent import Proposal
from automl.model import Model
```

**Noun classes + session machinery.** No domain-verbs at the top (those live in their submodules).

**No parallel `AutoML(...).fit(X, y)` entry point.** The recipe-file (`projects/<name>/config.py`) is the only project definition; a sklearn-style constructor would compete with it.

## 13. Cross-cutting decisions

### 13.1 Schema location

Schemas live in the domain that owns the concept they describe.

| Concept | Schema location |
|---|---|
| `DataSpec`, `DataSource`, `DataPipeline`, `Dataset` identity, `TrialDataContract` | `data/` |
| `Task`, `RunConfig`, `Splits`, `ModelsConfig` | `project/` |
| `BaseModel`, model signature | `model/` |
| `EvalSpec`, `Metric`, `EvalDataset`, `EvalResult`, `EvalIndex`, `Predictions` | `eval/` |
| `Proposal` contract + field constants | `agent/proposal.py` |
| `TrialSummary`, `TrialDetails`, `ParentExperimentRef` | `trial/types.py` (seam returns these) |
| `LeaderboardData`, `ComparisonResult`, `MetricDelta` | `experiment/views/types.py` |
| `TrialMetadata`, `SeedSelection`, `TimingReport`, `TrialManifest` | `trial/metadata.py` |
| `FeatureRegistry` | `data/features.py` |
| `Issue`, `ValidationReport`, `Severity` | `validate/base.py` (`CheckSpec` deleted — sub-spec 04 Q1; `ValidationReport` gains `schema_version: int = 1` + `from_dict`) |

**Artifact-schema rule:** the *type* lives in the domain that owns the concept; the *writer* lives in `mlflow/artifacts/`. The writer imports the type, validates input against it, persists.

### 13.2 Validate distribution

Framework in `validate/`; checks in `<domain>/checks.py`. See §9.2.

### 13.3 Runner extension shape

Today: one runner (`runner/trial.py`) as a straight-line procedure — no stage abstraction. When a second runner (HPO, feature ablation, data-ratio sweep) actually needs to exist, *that's the moment* to refactor: extract shared primitives into `runner/_stages.py` and add `runner/<new>.py` as a sibling. **No speculative central dispatcher, no preemptive stage abstraction.** Per `feedback_extension_points_follow_demand`.

### 13.4 MLflow seam invariant

**Domain code never `import mlflow`.** Always `automl.mlflow.<noun>`. Readers return typed domain objects. See §9.1.

### 13.5 Project context threading — deferred to a dedicated sub-spec (HIGHEST PRIORITY)

This is the most important interface decision still open, and it deserves its own sub-spec session before any implementation begins. The structural spec fixes only that **project context is the cross-cutting ambient state every domain reads** — the *mechanism* is open.

**Scope the sub-spec must resolve:**

- **What "project context" contains, and how it varies by call site.** Some library entries need only a project *name* (the exploration phase, before `config.py` is written). Some need the loaded `config.py` and all its typed fields (the runtime path). Some need a subset (env, MLflow URI, GCS bucket). The sub-spec must enumerate which call sites need which facets.
- **The exploration-phase scenario.** Notebook-1 in `projects/<name>/notebooks/`: the DS has a project folder (and thus a name) but no `config.py` yet. The library must let them poke at data, sketch sources, run quick checks — without crashing on config-load. The sub-spec must decide how `use_project` behaves when `config.py` is absent or partial.
- **Whether `config.py` is the unified source of project context** ("front config" pattern) — i.e. one file declares everything the framework needs and every module reads from it. The sub-spec must decide whether this becomes the canonical contract.
- **Real-time overrides.** CLI flags like `--dry-run`, `--experiment`, `--project-root` must override recipe values at invocation time without mutating the recipe. The sub-spec must define the override layer (where overrides are applied, how they're discovered).
- **The per-function threading rule.** Does each public function accept `ctx: ProjectContext` explicitly? Pull from contextvar implicitly? Both? The rule must be one a contributor can apply mechanically, so the codebase doesn't drift to half-implicit / half-explicit again.
- **Multi-project scenarios.** Concurrent agent runs, tests, tools that compare across projects.

**What's locked in this structural spec:**

- Project context lives in `project/context.py`.
- The active-project state is a `contextvar`.
- `use_project(name)` is a Tier-1 facade verb.
- The `ProjectContext` type is a domain type in `project/`.

**What's NOT locked here** (goes in the sub-spec):

- Whether each function signature takes `ctx: ProjectContext | None = None` or pulls from `contextvar` or both.
- How exploration-phase context (name only, no config) is represented.
- How CLI flag overrides are layered on top of `config.py` values.
- What `ProjectContext` actually contains as fields.

### 13.6 Feature engineering

`data/features.py` for the `FeatureRegistry`. Sub-folder (`data/features/`) only if structural-learning workflows (feature ablation, versioned transformers) become real. Per `feedback_extension_points_follow_demand` — no preemptive expansion.

### 13.7 Test layout

Tests live in a dedicated top-level `tests/` tree (NOT co-located inside `automl/<domain>/tests/`). The tier separation we use today is preserved; within each tier, the structure mirrors the domain folders for discoverability.

```
tests/
├── unit/                       ← function- and class-level tests; no live services
│   ├── project/
│   ├── data/
│   ├── model/
│   ├── eval/
│   ├── runner/
│   ├── experiment/
│   ├── mlflow/
│   ├── validate/
│   └── utils/
├── integration/                ← multi-domain tests; may use real MLflow (file-backed)
│   ├── data_pipeline/
│   ├── runner_full/
│   ├── experiment_views/
│   └── ...                     ← structured by the scenario being tested, not by domain
├── contracts/                  ← ratchet tests pinning architectural invariants (flat, by topic)
├── regression/                 ← golden-output manifests
├── e2e/                        ← full-stack against live services
└── shared/                     ← pytest fixtures and helpers
```

Notes:

- **Unit tier mirrors domains 1:1.** When you change `automl/data/pipeline.py`, you know to look in `tests/unit/data/` for tests of it.
- **Integration tier is organized by scenario, not by domain.** Integration tests cross domains by definition; forcing them into one domain folder hides what they actually test.
- **Contract tier stays flat.** Contract tests pin invariants of the package shape itself — they're orthogonal to domains.
- **QA tier dropped** (was: `tests/qa/`). After review, QA-grade smoke tests overlap with `e2e/` and don't justify their own tier. Anything we'd put there goes into `e2e/` with appropriate gating.
- **Test pruning** of the ~992 current tests is the implementation plan's job — not this spec's. The structural spec only fixes *where* tests live, not *which* tests survive.

**Known ratchet test that breaks on this change:** `tests/contracts/test_pytest_structure.py` currently asserts `testpaths == ["tests/unit", "tests/contracts"]`. The new layout adds `integration/`, `regression/`, `e2e/`, `shared/` as peers. The new `pyproject.toml`'s `testpaths` will differ, and this contract test must be updated in the same commit that introduces the new test layout. Treat the contract update as part of the layout change, not as a follow-up.

### 13.8 Dataset / EvalDataset shared seam

Training Dataset (`data/dataset.py`) and EvalDataset (`eval/eval_dataset.py`) are conceptually related — both are immutable, content-hashed, parquet-on-GCS artifacts — but their identity shapes diverge:

- **Dataset identity** composes `source_identity_hash` + `feature_registry_hash` + `data_content_hash` + `schema_hash`. The training-data view is a function of *what came from where with what features* — the source and the feature set are part of the contract.
- **EvalDataset identity** (settled in sub-spec 07): `external` composes `content_hash` + `schema_hash` + recipe hash (it owns bytes — content is its identity); `split_view` is **recipe-only** — `of_dataset_id` (content-addressed link to a training Dataset) + `split_id_col` + `buckets`, with no stored content/schema hash. A `split_view` is a *view*: it delegates bucket realization to `data.load_dataset_by_id(of_dataset_id, split_range=buckets)` and gets integrity from the content-addressed id + data's L2 load-time validation. Eval views can be a partition *of* a Dataset (`split_view`), a wholly separate dataset for re-eval (`external`), or an augmented variant (column-add via `Augmentation`, which preserves the parent EvalDataset's identity).

The two classes stay in their own domains. **No top-level `dataset/` or `snapshot/` domain.** The Six Nouns vocabulary (§5) is preserved — adding a Snapshot folder would re-introduce exactly the term §5 deliberately retired.

**Unification checkpoint (sub-spec 07 resolution).** 07 examined whether to unify the two identity families and chose *not* to — but recorded the clean target for when it earns its place. The north star: one content-addressed-table substrate + a composable **lineage** descriptor (`MaterializedFromSource` / `SliceOf` / `ExternalImport` / `AugmentationOf`) + train-vs-eval as a *consumption role*, under which `split_view` dissolves into "consume a `SliceOf` in the eval role." Not built now: it re-introduces the retired noun and rewrites the working `evaluate()` caching model with no current functional pain. The watched **tripwire** is the `eval → data` runtime-loading seam (§8.4): if eval grows to depend on progressively more of data's loading internals, or a third byte-owning artifact family appears, re-open this. 07 took the *lighter* slice — `split_view` delegating realization to data's slice machinery — which removes the duplicated bucket-realization (the one trigger that had landed) and pre-pays the hard part of any future unification.

**What's shared lives in `utils/hashing.py`:**

| Primitive | Used by | Why utility |
|---|---|---|
| `dataframe_content_hash(df)` | data, eval | Hashing a DataFrame's contents has no AutoML semantics. |
| `schema_hash(df)` | data, eval | Hashing column-names+dtypes has no AutoML semantics. |
| `json_hash(value)` | data, eval | Deterministic JSON-serializable hash; no AutoML semantics. |

These were `_json_hash`, `dataframe_content_hash`, `schema_hash` in legacy `data/snapshot.py` — the first underscore-prefixed and reached for across a domain boundary by `eval/snapshot.py`. Promoting them to a public utility removes the smell at the source.

**What stays in `data/split.py`:** `HashKey` and `hash_key_columns` (deterministic per-row hashing for split assignment). These are tightly coupled to the splitting model — a `split_view` EvalDataset legitimately imports them from `data.split` because the relationship "eval split is derived from training Dataset's split" is real. They're not AutoML-agnostic; they encode the split contract.

**Dependency direction:**

- `eval → data` is allowed for concept-level needs (`Dataset` type for linkage; `DataPipeline` for materialization; `HashKey` / `hash_key_columns` from `data.split`; GCS paths via `Dataset` properties). Content-hash primitives come from `utils/hashing.py`, not `data/`. See §8.4 for the enumerated list.
- `data → eval` does not exist and must not be introduced — training-Dataset materialization predates and is independent of any eval view.
- **No domain imports another domain's underscore-prefixed privates.** When a private helper needs to cross a domain boundary, it gets promoted: to `utils/` if AutoML-agnostic, or to the owning domain's public API if domain-meaningful. This rule applies symmetrically to the other current leak (`eval/loading.py` imports `_load_snapshot_by_id` from `data.snapshots`) — sub-spec 05 (Data) lands the public registry API (`load_dataset_by_id`) and sub-spec 07 consumes it, retiring the private import.

**Forward-looking guidance.** If new artifact families with their own identity appear (calibration dataset, drift baseline, prediction-set with cross-experiment identity), the same pattern applies:

1. The identity class lives in the owning domain.
2. AutoML-agnostic hash primitives extend `utils/hashing.py` as sibling functions.
3. Domain-meaningful primitives (e.g. a future "fingerprint how a feature transform was applied") live in the owning domain's `*.py` file and are imported by other domains as public symbols if needed.

This keeps the layout stable: artifact families multiply without growing a separate "identities" folder, and the seam between `data/` and `eval/` (or any future analogous pair) is a known surface rather than a tangle.

## 14. Decision rules — "where does this go?"

When a new piece of code shows up, run it through the table. If two domains both answer yes, that's a coupling to untangle.

| Layer / domain | Test |
|---|---|
| **project** | Does it set or read "which project am I in / what's its config"? |
| **data** | Does it produce, store, or describe training data? |
| **model** | Does it define what a Model IS, how it's packaged, or how it's loaded? |
| **eval** | Does it score model outputs against ground truth? |
| **runner** | Does it execute a chain (data → fit → eval → log or similar) for one trial? |
| **experiment** | Does it organize, persist, query, or display the collection of trials? |
| **mlflow** | Does it call the MLflow Python API directly? (If yes, it goes here regardless of which noun it's about.) |
| **validate** | Does it produce a `ValidationReport`? (The framework, not the checks — checks go in the domain.) |
| **utils** | Has zero AutoML semantics — would make sense in a random Python project? |
| **cli** | Is it an argparse entry point? |

## 15. What this spec defers — sub-specs and implementation plan

The structural spec settles vocabulary, layers, folder shape, and cross-cutting rules. Several interface-level decisions still need their own sessions before code is written; others can be resolved by the implementation plan as it goes.

### 15.1 Sub-specs needed BEFORE implementation begins

These are interface-level decisions that, if left to the implementation plan, will get made under pressure and ossify wrongly. Each gets its own focused session (3–5 questions, much shorter than this structural spec).

**Priority 1 — Project context threading.** See §13.5 for full scope. The most important sub-spec: defines what `ProjectContext` contains, how the contextvar+explicit dual pattern works, how the exploration-phase notebook scenario is supported, how CLI overrides are layered. *Nothing in the library can be implemented uniformly without this.*

**Priority 2 — MLflow seam interfaces.** Define every function signature in the `mlflow/{project,experiment,trial}/` per-noun folders and the typed return shapes (`TrialSummary`, `LeaderboardData`, etc.). Locks the seam that the rest of the library depends on. Defines the artifact-writer contracts in `mlflow/project/artifacts.py` + `mlflow/trial/artifacts/` (see sub-spec 02 §4).

**Priority 3 — Cleanup orchestration.** Defines the cascading delete order (MLflow runs → MLflow experiment → GCS prefix → local trial dirs) at each scope, the dry-run semantics, the error-handling per step. Today's `cleanup.py` is load-bearing; splitting it without this sub-spec risks orphaned blobs.

*(Previously planned sub-specs no longer needed:)*

- *Runner stage contract* — stages are not an abstraction today (§13.3).
- *Agent timeline contract* — `hooks/agent_timeline.py` behavior is already established; the refactor moves the file and updates imports without changing behavior. This is implementation-plan work, not a design question.
- *CLI verb sketch* — promoted out of sub-specs and into the structural spec (§11.1 below) as a high-level design section.

### 15.2 Resolved by the implementation plan

- **Migration sequencing.** Branch-and-cutover plan, what gets ported first, acceptance gate.
- **Test pruning.** Of the ~992 current tests: which transfer 1:1, which collapse, which become contract tests, which delete. Per-tier analysis.
- **Tier 1 facade exact exports.** §13 sketches; the final list comes after per-domain `__init__.py` files are written.
- **Per-file move list.** §7 shows where new files go; full mapping from each legacy file is in the plan.
- **Project-local subclass resolution.** How `projects/<name>/data/pipeline.py` gets wired in as a `DataPipeline` subclass.
- **Logging conventions.** Level, format, who calls `configure_logging`.
- **Skill + agent file adjustments.** SKILL.md prose, agent prompts, hooks — adjust as CLI verbs settle.

## 16. Inspiration — what the closest analogs say

- **sklearn:** domain folders (`linear_model/`, `cluster/`, `metrics/`), small top-level facade, Tier-3 anchors in domain (`sklearn.base`), validation utilities in `utils/`. Schemas live where they're used. ✓ matches our shape.
- **pytorch:** `nn/` with Module ABC + builtins; `optim/` with Optimizer ABC; `utils/data` for datasets. ✓ matches our domain-with-ABC pattern.
- **optuna:** closest cousin. `study/`, `trial/`, `samplers/`, `storages/` (persistence as its own top-level), `visualization/`. ✓ matches our four-layer shape — including the Experiment/Trial *separation*: optuna keeps `study/` and `trial/` as peers, and we now do the same (`experiment/` + `trial/` as peer domains — see §8.6 decomposition note and §17.11).

---

## 17. Forward-Looking Guidance (NON-BINDING)

This section captures discussions about where future capabilities **might** live if the user chooses to invest in them. **Nothing here is committed.** Any of these decisions can be revisited; placements may shift based on what actual demand looks like. Implementation of any item requires explicit user confirmation before starting.

The purpose is to give a *first guess* placement so that the next time we have a "where would X go?" conversation, we start from a position rooted in the conceptual model — not so future work follows the placements automatically.

### 17.1 Feature engineering expansion

If feature ablation experiments, versioned transformers, or a feature store become real workflows:
`data/features.py` → `data/features/` folder with one file per kind (`registry.py`, `transformers.py`, `ablation.py`, etc.). Same pattern as `data/sources/`.

### 17.2 Additional runner types

If HPO sweeps, feature-ablation runs, data-ratio sweeps, or ensemble runs become first-class workflows:
Sibling files in `runner/` — `runner/hpo.py`, `runner/feature_ablation.py`, `runner/data_ratio.py`, `runner/ensemble.py`. Each composes the same `runner/stages/` primitives differently. **No central dispatcher; no runner-type registry.** Each gets its own CLI verb when promoted.

### 17.3 Stage interface as a Tier 3 contract

If a real demand emerges for projects to inject custom stages (e.g., a project-specific data-load step) into the chain:
Promote the stage interface to an ABC: `runner/stages/base.py` with a `Stage` contract. Until then, stages remain concrete functions composed by name.

### 17.4 Model lifecycle expansion

`model/` is thin today (`base.py`, `packaging.py`, `checks.py`). As serving manifest, signature inference, model versioning, or model-card generation become real concerns:
New sibling files under `model/` — `model/serving.py`, `model/signature.py`, `model/versioning.py`, `model/card.py`. The `BaseModel` ABC likely grows additional optional methods.

### 17.5 Re-evaluation workflows

Today re-eval is a single CLI verb. If scheduled or batched re-eval becomes a workflow (e.g., monthly re-eval of all promoted models against new data periods):
`eval/reeval.py` as a sibling to `eval/evaluate.py`. Possibly an `automl reeval` verb that loops over runs.

### 17.6 Experiment-level Dataset views

Today a Dataset is shared across experiments by content hash. If experiments need their own augmented or filtered Datasets:
A per-experiment dataset pin lives on the Experiment (already supported via the experiment-overview run's active-dataset tag). A *derived* Dataset (filter / augment / subset) would appear as `data/derived.py` or as a method on `Experiment`.

### 17.7 Additional inspection views

Beyond pairwise `compare`:
A/B serving comparison, drift monitoring, calibration plots, threshold sensitivity views grow under `experiment/views/` as sibling files (`drift.py`, `calibration.py`, `serving_ab.py`).

### 17.8 Per-domain check growth

If `<domain>/checks.py` exceeds ~300 lines or holds >5 unrelated check families:
Split into `<domain>/checks/` folder with one file per family. The orchestrator's imports point at the new folder; no other behavior change (there is no registry — it was removed in sub-spec 04 Q1; orchestrators call check functions directly).

### 17.9 Additional data sources

New ingestion paths: `data/sources/<name>.py` implementing `DataSource`. Likely candidates: REST API source, on-disk parquet source, BigQuery source. Each ships as a Tier 3 implementation.

### 17.10 CLI verb sub-grouping

As CLI surface grows, group by user intent — `automl trial <subverb>`, `automl experiment <subverb>`, `automl validate <target>`. **Don't leak library structure into verb names** (no `automl experiment.views.leaderboard`).

### 17.11 Agent-loop extensibility (the two latent axes)

`agent/` ships in v1 as a **straight relocation** of today's proposer↔coder loop (§8.8) — no extension machinery, per `feedback_extension_points_follow_demand`. But two extension axes were identified and are recorded here so the boundary work isn't lost:

- **More agents / roles.** Today there are two roles (proposer, coder), defined by `agents/*.md` (prompt + tools + model routing). Future roles — an issue-inspector, a data-explorer, etc. — would add a definition and wire it in. The clean unit is "an agent definition"; this is the closest analog to eval's "subclass `Metric`" extensibility.
- **More drivers.** The loop is launched as a `claude` subprocess today. A second driver (e.g. `codex`) would be a localized extraction from `agent/launch.py` — which is *why* the claude-specific invocation building is kept in one place behind a single `build_launch`.

**Neither is built now** (one driver, two roles, no second consumer). When a real second agent or driver appears, that is the moment to design the seam (a `Driver` protocol and/or a typed `Agent` definition) — a dedicated pass, not speculative structure carried in v1. The loop's *sequencing* stays LLM-driven regardless.

### 17.12 Things considered and explicitly NOT adopted

These came up in the structural-design conversation and were rejected for stated reasons:

- **A sklearn-style `AutoML(...).fit(X, y)` Python entry point.** Would compete with the recipe-file model. See §12.
- **A central runner-type registry.** Future runner types are sibling files, not registered plugins. See §13.3.
- **A central `schemas/` folder.** Schemas live with the domain that owns the concept. See §13.1.
- **A "Study" rename of Experiment.** "Experiment" matches MLflow's term and is the user's natural language. See §5.
- **A `proposal/` top-level domain.** Proposal has no folder of its own — the contract folds under `agent/proposal.py` (it is the proposer↔coder handoff) and validation under per-domain checks. See §8.8.

**REVERSED (now adopted) — `trial/` as a top-level domain.** This list previously rejected "a `trial/` top-level domain (peer to `experiment/`)," on the reasoning that there is no trial without an experiment. That call was reversed during sub-spec 09: the parent↔child *relationship* is real, but it is carried by MLflow tags + seam routing, not by folder nesting — and the old `experiment/` had become a muddy catch-all (Trial ops + Proposal + views + the agent loop all in one folder). MLflow itself keeps Experiment and Run separate, and optuna keeps `study/` / `trial/` as peers (§16). So `trial/` is now a peer domain, and the agent loop was extracted into `agent/` at the same time. See §8.6 decomposition note, §8.7, §8.8.

Anything still on this list that becomes desired later requires re-opening the structural design — not adding the rejected piece on the side.

---

## Appendix A — Mapping legacy → new (orientation, not the migration plan)

The implementation plan will contain the full file-by-file move list. This appendix is a high-altitude orientation only.

| Legacy folder | New home |
|---|---|
| `automl/core/` | Dissolved: `project/`, `model/`, `data/features.py` |
| `automl/cleanup.py` | `project/cleanup.py` (cascading at all three scopes; see §8.1) |
| `automl/data/` (snapshot, snapshots, run_snapshot, loader) | `data/dataset.py`, `data/registry.py`, `data/pipeline.py` |
| `automl/data/snapshot.py` private hash helpers (`_json_hash`, `dataframe_content_hash`, `schema_hash`) | `utils/hashing.py` (promoted to public: `json_hash`, `dataframe_content_hash`, `schema_hash`). See §13.8. |
| `automl/data/adapters/` | **Deleted** — legacy `*Pipeline` constructor-sugar wrappers (sub-spec 05 Q2). The real source classes already live in `data/sources/`. |
| `automl/eval/snapshot.py` | `eval/eval_dataset.py` |
| `automl/eval/loader.py + loading.py` | `eval/_load.py` |
| `automl/eval/publish.py` | `eval/prepare.py` |
| `automl/inspect/` | Split: `leaderboard` / `compare` → `experiment/views/`; `show_trial` / `load_model` → `trial/show.py`; `experiments` → `mlflow.project.list_experiments` (seam); `load_data_snapshot` → `data.load_dataset_by_trial` |
| `automl/io/` | `utils/io/` |
| `automl/loop_context/` | `agent/proposer_context.py` (the proposer packet) + `experiment/views/queries.py` (`recent_failures`/`strategies_attempted`) + `experiment/views/summary.py`; raw searches → seam |
| `automl/mlflow/store.py` | Split across `mlflow/project/`, `mlflow/experiment/`, `mlflow/trial/` folders (incl. `mlflow/experiment/queries.py`). See sub-spec 02 §4. |
| `automl/mlflow/overview.py` | Collapsed into `mlflow/project/overview.py` and `mlflow/experiment/lifecycle.py` |
| `automl/profile/` (core.py + snapshot.py) | `data/profile.py` (single file; MLflow writing → `mlflow/project/artifacts.py`) per sub-spec 05 Q5 |
| `automl/propose/` | Contract + constants → `agent/proposal.py`; validation (`validate()`) → `agent/checks.py::proposal_schema` (per §9.2) |
| `automl/runner/_execute.py + _stages.py + template.py` | `runner/trial.py` (the chain) + cohesive modules (`paths`/`contract`/`validation`/`_pyfunc_check`/`manifest`/`_modules`/`session_lock`) + `runner/template.py`. **No `runner/stages/`** in v1 — the stage abstraction is the deferred Tier-2 north-star (sub-spec 08; §13.3/§17.2). |
| `automl/session/` | `runner/session_lock.py` |
| `automl/trial/{creation,fork,promotion,cleanup}.py` | `trial/{create,fork,promote,cleanup}.py` |
| `automl/trial/packaging.py` | `trial/packaging.py` (notebook→source extraction; 06 correction — *not* `model/`, whose `packaging.py` keeps only `save_model`) |
| `automl/cli/run_loop.py` (`build_launch`, `LaunchSpec`, …) | `agent/launch.py`; CLI verb `automl experiment run` → thin `cli/` wrapper |
| `automl/hooks/agent_timeline.py` (1955L) | `agent/timeline.py` (library) + thin `hooks/agent_timeline.py` stub; MLflow writes route through the seam |
| `automl/trial/run.py` | Collapsed (1-line wrapper) |
| `automl/utils/` (dormant) | Become real: `utils/logging.py` actually used; `utils/io/`, `utils/paths.py` added |
| `automl/validate/builtin/*` | Most moved to `<domain>/checks.py`; framework stays in `validate/` |
