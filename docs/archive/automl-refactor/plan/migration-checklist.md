# Migration Checklist — legacy `automl/` package → fresh `automl/` package

**Purpose:** track every public symbol in the legacy library and where it ends up in the new design. Used during migration to confirm coverage; used at end as audit ("did we leave anything behind?").

**Granularity:** tracks **public functions / classes / constants**, not files. A function that splits across two new files is still "covered" if both halves land. Private helpers (`_foo`) are listed only when they encode contract or are widely imported.

**Authority:** the structural spec (§7 + Appendix A) and per-domain sub-specs are the source of truth for *where* things go. This checklist tracks *whether* we've actually moved them.

**✅ Phase-0 completeness audit (2026-05-27).** Inventoried every public symbol in the real
legacy `automl/` tree (+ `hooks/`) across all areas (3 parallel audits) and cross-checked
against this ledger. **Verdict: coverage is complete** — every legacy public symbol has a
disposition (ported-to-`<home>` or `[-]` dropped). Exactly **one** gap was found and filled:
`PROJECT_NAME_RE` (`cli/project.py`) now has a row → `project/scaffold.py`. All other "stale /
missing" findings from the audit were **implementation-status** (legacy code still uses old
names / old structure because the migration hasn't run yet) — *not* ledger gaps; those flip
`[ ]`→`[x]` during execution. The earlier stale-row fixes (`routing.py`→`_routing.py`;
`gcs_paths` `run_mode` dropped + `route_namespace`→`namespace`) are already applied.

---

## Status legend

| Symbol | Meaning |
|---|---|
| `[ ]` | Mapping decided; not yet implemented in new tree |
| `[/]` | Partially migrated — some public symbols covered, others pending |
| `[x]` | Fully migrated — every public symbol from this current file has a new home AND code is written |
| `[-]` | Intentionally dropped (won't exist in new design; reason in Notes) |
| `[?]` | Mapping uncertain — needs decision in an upcoming sub-spec |

---

# Library — `automl/`

## Top-level

### `automl/__init__.py`
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `_load_dotenv_from_cwd` | helper | `project/config.py::_load_env` | `[x]` | Replaced by `python-dotenv`-backed project-load env handling. |
| `_read_simple_dotenv` | helper | DROP | `[-]` | Hand parser dropped; `python-dotenv` owns parsing. |
| (re-export) `use_project` | facade | `automl.use_project` (kept; new semantics per sub-spec §4.1) | `[x]` | Phase 1 facade landed. |
| (re-export) `load_project` | facade | `automl.use_project` (unified) | `[-]` | Same function; dropped name |
| (re-export) `clear_project` | facade | `automl.clear_session` (renamed per sub-spec §4.4) | `[-]` | Renamed |
| (new) `session`, `active_session`, `update_session`, `clear_session` | facade | `automl.<name>` per sub-spec §4 | `[x]` | Phase 1 facade landed. |
| (new) `Session`, `ProjectConfig` | facade | `automl.<name>` (re-exports) | `[x]` | Phase 1 facade landed. |
| (new) `Dataset`, `Experiment`, `Proposal`, `Model` | facade | `automl.<name>` (re-exports) | `[x]` | Phase 7 facade re-exports landed; `Model` aliases `BaseModel`. |

### `automl/errors.py`
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `AutoMLError` | class | `errors.py` (top-level) | `[x]` | |
| `ValidationError` | class | `errors.py` | `[x]` | Additive compatibility name; `ValidationReport.raise_if_failed` remains dropped. |
| `ContractError` | class | `errors.py` | `[x]` | |
| `ConfigError` | class | `errors.py` | `[x]` | |
| `ProposalError` | class | `errors.py` | `[x]` | |
| `TrialError` | class | `errors.py` | `[x]` | |
| `TrialFitError` | class | `errors.py` | `[x]` | |
| `TrialEvalError` | class | `errors.py` | `[x]` | |
| `ProjectError` | class | `errors.py` | `[x]` | |

### `automl/cleanup.py` (1087L)
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `RouteCleanupTarget` | dataclass | `project/cleanup.py` (folded into `CleanupPlan.mlflow_experiment_targets` + `gcs_prefix_patterns`) | `[x]` | Resolved by sub-spec 03 §12.5; landed in Phase 4 cleanup plan schema |
| `MlflowDeleteResult` | dataclass | `project/cleanup.py` (folded into `CleanupResult.mlflow_experiments`) | `[x]` | Resolved by sub-spec 03 §12.5; landed in Phase 4 cleanup result schema |
| `RunCleanupTarget` | dataclass | `project/cleanup.py` (folded into `CleanupPlan.mlflow_run_targets` + `gcs_prefix_patterns`) | `[x]` | Resolved by sub-spec 03 §12.5; landed in Phase 4 cleanup plan schema |
| `CleanupPlan` (legacy) | dataclass | `project/cleanup.py` (renamed + redesigned; new schema in sub-spec 03 §10) | `[x]` | Resolved by sub-spec 03 §10 + §12.5 |
| `build_cleanup_plan` | function | `project/cleanup.py::_build_plan` (private) | `[x]` | Resolved by sub-spec 03 §9 |
| `require_confirmation` | function | DROPPED — replaced by `--apply` gate per sub-spec 03 §5.2 | `[-]` | No interactive confirmation in new design |
| `apply_cleanup_plan` | function | `project/cleanup.py::_apply_plan` (private) | `[x]` | Resolved by sub-spec 03 §9 |
| `run`, `_build_parser` | argparse | `cli/project.py` + `cli/experiment.py` + `cli/trial.py` (per noun's delete subcommand) | `[x]` | Phase 6 noun delete wrappers cover cleanup argparse; no standalone cleanup parser remains. |
| `main` | argparse | DROPPED — no `__main__` block in library per §11.4 | `[-]` | |

---

## `automl/cli/`

| File | Verb | New home (CLI verb + library destination) | Status | Notes |
|---|---|---|---|---|
| `cli/__init__.py` | dispatcher | `cli/__init__.py` (noun-first root parser) | `[x]` | Phase 6 dispatcher with root session flags. |
| `cli/__main__.py` | entry | `cli/__main__.py` | `[x]` | Phase 6 entry point delegates to root parser. |
| `cli/cleanup.py` | `cleanup` | dissolved into noun delete verbs → `project.cleanup` | `[x]` | `project delete`, `experiment delete`, and `trial delete` wrap the cleanup engine. |
| `cli/eval.py` | `eval` | `cli/eval.py` → `eval.evaluate` + `eval.registry` | `[x]` | Phase 6 `eval list` / `eval compute`. |
| `cli/inspect.py` | `inspect` | dissolved into `experiment`/`trial` views | `[x]` | Retired top-level `inspect`; new verbs are `experiment list/leaderboard/compare/summary` and `trial show`. |
| `cli/loop_context.py` | `loop-context` | **module dissolved** — `for-proposer` → `experiment proposer-context` (`agent.proposer_context`); `leaderboard`/`summary` → `experiment …` (09); `show-trial` → `trial show` (10) | `[x]` | Phase 6 completes the catalog cleanup and retired-verb ratchet. |
| `cli/profile.py` | `profile` | `data profile` → `data.profile` | `[x]` | Retired top-level `profile`; Phase 6 `data profile` wraps the data-domain function. |
| `cli/project.py` | `project` | `cli/project.py` → `project.scaffold` + `project.metadata` + `project.dependencies` | `[x]` | Phase 6 moved project creation into `project/scaffold.py` and added metadata/deps wrappers. |
| `cli/project.py::PROJECT_NAME_RE` | const | `project/scaffold.py` | `[x]` | Project-scoped lower-snake validation regex moved with scaffold creation. |
| `cli/propose.py` | `propose` | DROP — verb collapses into `cli/validate.py`'s `proposal` sub-verb | `[-]` | Sub-spec 04 Q5; `--output` flag preserved by adding it to `validate proposal` |
| `cli/run_loop.py` | `experiment run` | `agent/launch.py` (`build_launch`, `LaunchSpec`, `ClaudeRole`) + thin `cli/` wrapper that executes | `[x]` | Phase 5: session model routing, agent frontmatter parsing, `AUTOML_PROJECT`/`AUTOML_EXPERIMENT_ID`/`AUTOML_NAMESPACE`/`AUTOML_INHERIT_DRY_RUN` transport, and A5 CLI wrapper covered by unit + e2e gates. |
| `cli/trial.py` | `trial` | `cli/trial.py` → `trial.*`; `trial run` → `runner.run_trial`; `trial lock` → `runner/session_lock.py` | `[x]` | Phase 7 adds authoring verbs create/fork/promote; list/run/show/delete/lock remain green. |
| `cli/validate.py` | `validate` | `cli/validate.py` → `validate.targets.*` (3 sub-verbs only: project, model, proposal) | `[x]` | Phase 6 completes `validate project`, `validate model`, and inferred/explicit-session `validate proposal`. |

---

## `automl/core/` (dissolves into multiple new domains)

### `core/__init__.py`
| Symbol | Kind | New home | Status |
|---|---|---|---|
| (empty re-exports) | — | — | `[-]` |

### `core/base_model.py`
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `BaseModel` | ABC | `model/base.py` | `[x]` | THE Tier 3 anchor. Sub-spec 06: +`required_transformer_entries(session=None)` hook. Phase 2 wires it to `RequiredTransformer` declarations. |
| (new) `RequiredTransformer`, `SklearnTransformer`, `describe_required_transformers` | dataclass/protocol/fn | `model/preprocessing.py` | `[x]` | Sub-spec 06 Q2/Q5/Q7 — project-mandated preprocessing contract |
| (new) `save_model` | function | `model/packaging.py` | `[x]` | Sub-spec 06 finding 3 — cloudpickle dump extracted from `runner/_execute.py:710`; path-based load deferred |
| (new) `check_required_transformers` | check | `model/checks.py` | `[x]` | Sub-spec 06 Q3.3 — framework-owned gate (reads `session.config.required_transformers`) |

### `core/config.py`
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `load_env` | function | `project/config.py` private helper | `[x]` | Implemented as `_load_env(repo_root)` inside `ProjectConfig.load()`. |
| `find_project_root` | function | `project/metadata.py` | `[x]` | Implemented as `find_repo_root()`. |

### `core/dependencies.py`
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `parse_dependency_name` | function | `project/dependencies.py` | `[x]` | |
| `allowed_dependencies` | function | `project/dependencies.py` | `[x]` | Exposed through `automl project deps`. |
| `main` (argparse) | function | `cli/project.py` | `[x]` | `python -m automl.core.dependencies` removed per §11.4; use `automl project deps`. |

### `core/feature_registry.py`
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `FeatureEntry` | dataclass | `data/features.py` | `[x]` | |
| `FeatureRegistry` | class | `data/features.py` | `[x]` | |

### `core/project_context.py` (732L — dissolves into project/config.py + project/session.py)
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `PROJECTS_DIR`, `PROJECT_FILENAME`, `INSTRUCTIONS_FILENAME`, `PACKAGE_COMPONENT_RE` | constants | `project/config.py` / `project/_import.py` | `[x]` | |
| `ProjectContext` | class | **REPLACED** by `ProjectConfig` + `Session` (sub-spec §3) | `[-]` | Major rename + split |
| `infer_project_name` | function | `project/metadata.py` | `[x]` | |
| `find_repo_root` | function | `project/metadata.py` | `[x]` | |
| `set_active_project` | function | **REPLACED** by `use_project` (sub-spec §4.1) | `[-]` | |
| `get_active_project` | function | **REPLACED** by `session` (sub-spec §4.2) | `[-]` | |
| `clear_active_project` | function | **REPLACED** by `clear_session` (sub-spec §4.4) | `[-]` | |
| `active_project` (cm) | context mgr | **REPLACED** by `active_session` (sub-spec §4.3) | `[-]` | |
| `cwd_project` | function | `project/metadata.py::infer_project_name` | `[x]` | Collapsed into the single inference helper. |
| `default_project` | function | `project/metadata.py::infer_project_name` | `[x]` | Collapsed into the single inference helper. |
| `current_project` | function | `project/session.py::session` | `[x]` | Ambient current project is now `Session`. |
| `project_metadata` | function | `project/metadata.py` | `[x]` | |
| `resolve_project_context` | function | `project/config.py` (`ProjectConfig.load` internal helper) | `[x]` | Unified into `ProjectConfig.load()` plus metadata/session bootstrap. |
| `load_project` | function | **REPLACED** by `use_project` (sub-spec §4.1) | `[-]` | |
| `project_context` (cm) | context mgr | `project/session.py::active_session` | `[-]` | Dropped as a duplicate name; `active_session` is the kept context manager. |

### `core/run_config.py`
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `SAFE_ROUTE_COMPONENT_RE` | const | `project/run_config.py` | `[x]` | |
| `ALLOWED_EFFORTS` | const | `project/run_config.py` | `[x]` | |
| `Split` | dataclass | `project/run_config.py` | `[-]` | Renamed to `Splits`; legacy `split=` keyword remains as compatibility alias. |
| `ModelRoute` | dataclass | `project/run_config.py` | `[x]` | |
| `ModelsConfig` | dataclass | `project/run_config.py` | `[x]` | |
| `RunConfig` | dataclass | `project/run_config.py` | `[x]` | |

### `core/task.py`
| Symbol | Kind | New home | Status |
|---|---|---|---|
| `BinaryClassification` | dataclass | `project/task.py` | `[x]` |
| `Regression` | dataclass | `project/task.py` | `[x]` |
| `Multiclass` | dataclass | `project/task.py` | `[x]` |

---

## `automl/data/`

**Mappings settled by sub-spec 05 (2026-05-24).** All data-domain `[?]` / `[/]` rows were resolved
before implementation. Final statuses below reflect the implemented/dropped outcomes. Renames
were driven by Q1 (snapshot→dataset retirement) + Q9 (RunDataContract→TrialDataContract).

### `data/__init__.py`
| Symbol | Kind | New home | Status |
|---|---|---|---|
| (re-exports) | facade | `data/__init__.py` (Tier 2 verbs + types per sub-spec 05 §5) | `[x]` |

### `data/adapters/` (three adapter pipelines)
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `GCSParquetPipeline`, `LocalCSVPipeline`, `SnowflakePipeline` | classes | DROP | `[-]` | Legacy `*Pipeline` constructor-sugar wrappers (no real overrides). Sub-spec 05 Q2 deletes the folder; users construct `DataSpec(source=…Source(...))` directly. |

### `data/contract.py`
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `SCHEMA_VERSION` | const | `data/contract.py` (per-type `schema_version: int = 1`) | `[x]` | Per-type schema versions landed for Phase 1 contract types. |
| `Shape`, `SplitShape` | dataclasses | DROP | `[-]` | Flattened into `DatasetRef.n_rows/n_columns` + `SliceContract.n_rows` (sub-spec 05 Q9) |
| `RunRef`→`TrialRef`, `SnapshotRef`→`DatasetRef`, `SplitContract`→folded into `TrialDataContract.splits` + `SliceContract`, `RunDataContract`→`TrialDataContract` | dataclasses | `data/contract.py` | `[x]` | Renamed + generalized to any named slices (Q9). Keep both `TrialRef.trial_id` (`<number>_<slug>`) and `TrialRef.run_id` (MLflow UUID); drop `SnapshotRef.prepare_event_id`, `SplitContract.view_hash`. |
| `validate_run_data_contract`→`validate_trial_data_contract` + new `validate_loaded_dataset` / `verify_loaded_slice` / `verify_trial_tag_lineage` | functions | `data/contract.py` | `[x]` | Four validators (L1–L4). Takes typed `Dataset`. |

### `data/loader.py`
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `build_pipeline` | function | `data/pipeline.py::build_dataset` / private pipeline construction | `[x]` | Slimmed orchestration reads `session.config.require_data_spec()` and builds `DataPipeline` directly. |

### `data/pipeline.py` (1262L)
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `gcs_blob_exists`, `write_df_to_gcs_as_parquet`, `write_df_to_gcs_as_csv`, `write_json_to_gcs`, `read_parquet_from_gcs`, `get_csv_from_gcs`, `get_json_from_gcs` | shim funcs | DROP — use `utils/io/gcs.py` directly | `[-]` | These are duplicated shims; collapse |
| `snowflake_session_cm` | shim | DROP — use `utils/io/snowflake.py` | `[-]` | |
| `DataPreview` | dataclass | DROP | `[-]` | No consumer in the new pipeline; preview output collapsed into dataset/profile artifacts. |
| `DataPipeline` | class | `data/pipeline.py` (slimmed; ctor → 3 args per Q7) | `[x]` | **Tier 3 anchor (orchestration-side)** per Q2. Phase 1 thin path landed; later breadth remains. |
| `read_sql_file` | function | DROP | `[-]` | Snowflake execution remains a source stub until live Snowflake support is implemented. |
| `get_df_from_snowflake` | function | DROP | `[-]` | Snowflake execution remains a source stub until live Snowflake support is implemented. |

### `data/prepare.py`
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `main` | argparse | `automl data materialize` CLI verb (structural §11.1 line 473) | `[x]` | Phase 6 adds `data materialize`; `python -m automl.data.prepare` remains removed. |

### `data/run_snapshot.py`
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `load_data_snapshot_for_run` → `load_dataset_by_trial` | function | `data/registry.py` | `[x]` | Resolves dataset_id + splits from trial contract (Q3/Q9). `_validate_run_lineage` → `verify_trial_tag_lineage`. |

### `data/snapshot.py` (518L)
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `FrozenDict`, `FrozenList` | utility | DROP | `[-]` | No public immutable container types in the new dataset schema. |
| `SnapshotIdentity` → `Dataset` + `ComponentHashes` | dataclass | `data/dataset.py` | `[x]` | Composite hash; field `snapshot_identity_hash`→`identity_hash` (Q1/Q4) |
| `LoadedDataSnapshot` → `LoadedDataset` (full) + `LoadedSlice` (one slice) | dataclass | `data/dataset.py` | `[x]` | RESOLVED (was `[?]`). No `df_train`/`df_test` — split-at-load (Q3/Q4). |
| `dataframe_content_hash`, `schema_hash`, `_json_hash`→`json_hash` | functions | **`utils/hashing.py` (PUBLIC)** | `[x]` | Promoted per §13.8 (fixes cross-domain-privates smell). `registry_content_hash` stays in `data/` (data-specific). |
| `compute_snapshot_identity` → `compute_dataset_identity` | function | `data/dataset.py` / `data.pipeline.build_dataset` | `[x]` | Identity construction is folded into the durable `Dataset` build path. |
| `SNAPSHOT_HASH8_RE`, `SNAPSHOT_NAME_RE` → `DATASET_HASH8_RE`, `DATASET_ID_RE` | constants | DROP/private | `[-]` | Public snapshot naming API retired; dataset ids are produced internally. |
| `snapshot_name`, `validate_snapshot_name` → `dataset_id` helpers | functions | DROP/private | `[-]` | Public snapshot naming API retired; dataset ids are produced internally. |
| `snapshot_gcs_paths` | function | DROP — replaced by `Dataset.data_gcs_uri`/`registry_gcs_uri`/`manifest_gcs_uri` properties | `[-]` | Q4 — paths are Dataset properties; eval uses them too. |
| `prepare_event_id` | function | DROP | `[-]` | Q4 — temporal audit covered by `Dataset.created_at` + trial start_time. |
| `build_data_manifest` | function | `data.dataset.Dataset.to_dict` / `data.pipeline.build_dataset` | `[x]` | Manifest construction is type-owned and emitted by the pipeline. |
| `split_to_manifest`, `split_from_manifest`, `apply_split_view`, `build_split_view` | functions | `data/split.py` / `data.registry` | `[x]` | Split view behavior is covered by `Splits`, `add_split_id`, and split-at-load registry paths. |
| `validate_data_manifest_v2`, `validate_split_view` | functions | `data/contract.py` (as `validate_loaded_dataset` / `verify_loaded_slice`) | `[x]` | |
| `validate_run_data_contract` (shim) | function | DROP duplicate — one canonical in `data/contract.py` | `[-]` | RESOLVED (was `[/]`) |
| `asdict_jsonable` | utility | DROP | `[-]` | Serialization is implemented per schema (`to_dict`) or through CLI `jsonable`. |

### `data/snapshots.py`
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `LoadedSnapshot` → `LoadedDataset` | dataclass | `data/dataset.py` | `[x]` | Unified with pipeline's loaded type (Q4) |
| `SnapshotSummary` → folded into `Dataset` | dataclass | `data/dataset.py` | `[x]` | DatasetIndex carries `Dataset` entries directly (Q4) |
| `SnapshotIndex` → `DatasetIndex` | dataclass | `data/dataset.py` | `[x]` | |
| `load_snapshot` → `load_dataset` / `load_dataset_by_id` | function | `data/registry.py` | `[x]` | Phase 1 single-range split path landed. |
| `list_snapshots` → `list_datasets` | function | `data/registry.py` | `[x]` | |

### `data/sources.py`
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `DataSource` | ABC | `data/sources/base.py` | `[x]` | Tier 3 anchor (source-side) |
| `SnowflakeSource` | class | `data/sources/snowflake.py` | `[x]` | Phase 2 resolvable stub; execution remains deferred. |
| `LocalCSVSource` | class | `data/sources/local_csv.py` | `[x]` | Phase 1 Home Credit harness source. |
| `GCSParquetSource` | class | `data/sources/gcs_parquet.py` | `[x]` | |

### `data/spec.py`
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `DataSpec` | dataclass | `data/spec.py` (kept) | `[x]` | `constant_drop_threshold` default 0.99→1.0 (bug fix, Q7); `pipeline_cls` kept (Q2) |

### `data/split.py`
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `hash_key_columns`, `add_split_id`, `hash_key_report`, `split_report` | functions | `data/split.py` (kept — hashing mechanism) | `[x]` | `hash_key_report` folded into `hash_key_columns` validation; `Splits` config type lives in `project/run_config.py` (Q8), NOT here |

### `automl/profile/` → `data/profile.py`
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `profile_active_snapshot`, `profile_snapshot` | functions | `data/profile.py` as `profile()` / `get_profile()` | `[x]` | Q5. Single file (not folder). |
| `ProfileResult` → `Profile` | dataclass | `data/profile.py` | `[x]` | URI-only; drops local Path (Q5) |
| `build_data_card`, `distill_observations`, `generate_deterministic_charts`, `write_*` | functions | `data/profile.py` (private check/chart lists) | `[x]` | Phase 2 keeps the check/chart lists private and thin. |
| MLflow publishing (today in `profile/snapshot.py`) | — | `mlflow/project/artifacts.py:write_profile/read_profile` | `[x]` | Moves out of data domain (§9.1); lands under project-overview run (Q5 bug-fix) |

### `core/feature_registry.py` → `data/features.py`
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `FeatureRegistry`, `FeatureEntry` | class/dataclass | `data/features.py` | `[x]` | Q6. Add `derived` flag + `source_columns` + `add_derived()`. |
| `golden`/`weak` flags, `apply_learning_flags`, `import_learning_flags` | fields/methods | DROP | `[-]` | Learning subsystem deferred per README; stays legacy (Q6) |

---

## `automl/eval/`

_Eval-domain rows updated by sub-spec 07 (Q1–Q6, 2026-05-24)._

### `eval/base.py`
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `is_scalar_value` | function | `eval/base.py` | `[x]` | cross-domain public (imported by `validate/builtin/contract_checks.py`) — keep |
| `Metric` | ABC | `eval/base.py` | `[x]` | carry forward with signing, aliases, and required augmentation metadata (Tier 3 anchor) |
| `EvalSpec` | dataclass | `eval/base.py` | `[x]` | primary/metrics, duplicate-name guard, required columns/augmentations, augmentation join, and scalar primary validation |
| `scalar_metric_records` | function | `eval/base.py` | `[x]` | cross-domain public — keep; excludes non-finite/non-scalar values |

### `eval/compatibility.py`
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `ColumnMissing`, `DtypeMismatch` | exceptions | DROP | `[-]` | Replaced by `EvalSpec.validate_columns()` and `ValidationReport` issues. |
| `check_model_eval_compatibility` | function | DROP | `[-]` | Compatibility is enforced by `EvalSpec.validate_columns()` and runner validation. |

### `eval/evaluate.py` (909L)
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `EvaluateResult` | dataclass | DROP → `EvalResult` (`eval/results.py`) | `[x]` | 07 Q6 — `EvaluateResult`+`report.json` consolidate into singular `EvalResult` |
| `evaluate` | function | `eval/evaluate.py` | `[x]` | 07 Q5 — `ctx→session`, `eval_snapshot_id→eval_dataset_id`; `_model`/`_model_feature_registry` injection params preserved; returns/persists `EvalResult`, `Predictions`, and `EvalIndex` |

### `eval/results.py` (NEW — 07 Q6)
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `EvalResult` | dataclass | `eval/results.py` | `[x]` | singular; `schema_version:1`+`from_dict`; `cached` runtime-only (excluded from to/from_dict); no `mlflow_url` field |
| `EvalIndex` | dataclass | `eval/results.py` | `[x]` | was `eval/manifest.json` TOC; `schema_version:1`+`from_dict` |
| `Predictions` | dataclass | `eval/results.py` | `[x]` | typed predictions artifact + sidecar manifest |

### `eval/loader.py` and `eval/loading.py`
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `load_evaluation_spec` (from `loader.py`) | function | DROP | `[-]` | `ProjectConfig.load()` eagerly imports `EVAL` from project config; no separate public loader remains. |
| `LoadedEvalSnapshot` (from `loading.py`) | dataclass | `eval/_load.py` (rename `LoadedEvalDataset`) | `[x]` | 07 Q3/Q6 |
| `load_eval_snapshot` (from `loading.py`) | function | `eval/_load.py` (rename `load_eval_dataset`) | `[x]` | Durable split-view and external branches; split-view delegates realization to `data.load_dataset_by_id` (07 Q1). |

### `eval/metrics.py`
| Symbol | Kind | New home | Status |
|---|---|---|---|
| `Auc`, `LogLoss`, `ThresholdSweep` | classes | `eval/metrics.py` | `[x]` |

### `eval/publish.py`
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `EvalSnapshotPointer` | dataclass | DROP → merged into `EvalDataset` (paths as properties) | `[-]` | 07 Q6 |
| `AugmentationPointer` | dataclass | DROP → merged into `Augmentation` | `[-]` | 07 Q6 |
| `prepare_eval_snapshot` | function | `eval/prepare.py` (rename `prepare_eval_dataset`) | `[x]` | Durable split-view and external publish path; returns `(EvalDataset, cached)`. |
| `prepare_eval_augmentation` | function | `eval/prepare.py` | `[x]` | Durable augmentation publish path; returns `(Augmentation, cached)`. |
| `prepare_eval_split_view` | function | `eval/prepare.py` | `[x]` | Recipe-only split-view path with durable manifest. |

### `eval/runner.py`
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `run` | function | `eval/evaluate.py` (rename `evaluate_frame`) | `[x]` | 07 Q5 — `eval/runner.py` deleted; name avoids `runner/` domain collision |

### `eval/snapshot.py` (826L)
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `EVAL_SNAPSHOT_HASH8_RE`, `EVAL_SNAPSHOT_NAME_RE`, `AUGMENTATION_NAME_RE` | constants | DROP / private helpers in `eval/eval_dataset.py` | `[-]` | Snapshot name/hash constants retired; augmentation name regex is private implementation detail. |
| `EvalSnapshotIdentity` | dataclass | `eval/eval_dataset.py` (rename: **`EvalDataset`**, absorbs `EvalSnapshotPointer`) | `[x]` | 07 Q6 — one type, paths as properties |
| `AugmentationIdentity` | dataclass | `eval/eval_dataset.py` (rename: **`Augmentation`**, absorbs `AugmentationPointer`) | `[x]` | 07 Q6 |
| `validate_eval_snapshot_name`, `validate_augmentation_name` | functions | DROP / private validation in `eval/eval_dataset.py` | `[-]` | No public eval-dataset name API; augmentation validation is enforced by constructors/prepare. |
| `eval_snapshot_gcs_paths`, `eval_augmentation_gcs_paths` | functions | DROP → `EvalDataset`/`Augmentation` GCS-URI properties | `[-]` | 07 Q6 (mirrors data §05 Q4) |
| `augmentation_identity_hash`, `compute_augmentation_identity`, `compute_eval_snapshot_identity` | functions | `eval/eval_dataset.py` (`compute_eval_dataset_identity`) | `[x]` | `augmentation_identity_hash` folded into `compute_augmentation_identity`; snapshot→dataset rename complete. |
| `build_augmentation_manifest`, `build_eval_manifest` | functions | `eval/eval_dataset.py` → `to_dict()` / `from_dict` | `[x]` | 07 Q6 — manifest is the serialized identity |
| `validate_eval_manifest_v1`, `validate_augmentation_manifest_v1` | functions | `eval/eval_dataset.py` / `eval/_load.py` validation | `[x]` | 07 Q2 — split_view no longer requires/stores realized `schema_hash`/`content_hash`; external load verifies manifest hashes. |

### Out-of-domain callers updated in lockstep (07 Q3/Q5/Q6)
| Caller | Change | Status |
|---|---|---|
| `cli/eval.py` | `--eval-snapshot-id`→`--eval-dataset-id`; `eval_snapshot_id`→`eval_dataset_id`; `mlflow_url` via mlflow-seam helper | `[x]` |
| `mlflow/artifacts/eval.py` | manifest key `eval_snapshot_id`→`eval_dataset_id` | `[x]` |
| `loop_context/proposer_packet.py` | reads `eval_dataset_id` from report | `[-]` | Per sub-spec 11, primary-eval row enrichment was dropped from proposer context; drill-down lives in `trial.show_trial`. |
| `validate/builtin/contract_checks.py` | imports module-shape checks from `eval/checks.py` | `[-]` | Legacy module-shape checks dropped with the old registry; `validate project` now checks the loaded recipe shape. |
| `runner/_execute.py` | consumes `EvalResult`; still passes `_model`/`_model_feature_registry` | `[x]` |
| `eval/prepare.py` GCS imports | `data.pipeline` shims → `utils.io.gcs` | `[x]` |
| `tests/contracts/test_eval_snapshot_layout.py`, `tests/unit/test_eval_public_surface.py`, `test_eval_snapshot.py`, `test_eval_publish.py` | repoint to new names/paths | `[x]` | Rebuilt as `tests/unit/eval`, `tests/integration/eval`, and contracts/e2e coverage rather than one-to-one file moves. |

---

## `automl/inspect/`

### `inspect/views.py`
| Symbol | Kind | New home | Status |
|---|---|---|---|
| `LeaderboardRow` | dataclass | **DROP** — replaced by `LeaderboardData` (`experiment/views/types.py`) with `rows: list[TrialSummary]`; `mlflow_url` derived via seam `run_url`, not stored | `[-]` | sub-spec 09 §Q6/§8.1 |
| `leaderboard` | function | `experiment/views/leaderboard.py` | `[x]` |
| `show_trial` | function | `trial/show.py` (sub-spec 10 — per-trial read) | `[x]` | Home is `trial/` post-decomposition; enriched as `TrialDetails` + eval results |
| `compare` | function | `experiment/views/compare.py` | `[x]` | composes `trial.show_trial` per run |
| `experiments` | function | `experiment/views/summary.py::experiments` (enriched view over seam `mlflow.project.list_experiments`) | `[x]` | CLI `experiment list` → the enriched view (00 §11.1 carry-back); seam returns logical ids |
| `load_model` | function | `trial/show.py` (sub-spec 10) | `[x]` | **Sub-spec 06 correction:** MLflow PyFunc load by run_id + project resolution — depends on mlflow seam; model domain (deps=`errors` only) cannot host it. Was wrongly mapped to `model/packaging.py`; home is `trial/` post-decomposition |
| `load_data_snapshot` | function | `data/registry.py::load_dataset_by_trial` | `[x]` |

---

## `automl/io/`

### `io/gcs.py`
| Symbol | Kind | New home | Status |
|---|---|---|---|
| `gcs_delete_blob_if_exists`, `gcs_blob_exists`, `gcs_list_prefixes`, `gcs_list_blob_names`, `gcs_promote_blob` | functions | `utils/io/gcs.py` | `[x]` | Equivalent kept APIs: `delete_prefix`, `blob_exists`, `list_prefixes`, `list_blob_names`; one-off promote helper dropped. |
| `write_df_to_gcs_as_parquet`, `read_parquet_from_gcs`, `read_parquet_head_from_gcs` | functions | `utils/io/gcs.py` | `[x]` | Landed as `write_parquet`/`read_parquet`; dedicated head helper dropped because no caller needs it. |
| `write_df_to_gcs_as_csv`, `get_csv_from_gcs` | functions | `utils/io/gcs.py` | `[x]` | Landed as `write_csv`/`read_csv`. |
| `write_json_to_gcs`, `get_json_from_gcs` | functions | `utils/io/gcs.py` | `[x]` | Landed as `write_json`/`read_json`. |
| `get_file_from_gcs`, `write_bytes_to_gcs`, `put_file_to_gcs` | functions | `utils/io/gcs.py` | `[x]` | Landed as `read_bytes`/`write_bytes`; generic file upload helper dropped. |

### `io/snowflake.py`
| Symbol | Kind | New home | Status |
|---|---|---|---|
| `snowflake_session`, `snowflake_session_cm`, `get_snowflake_data` | functions | DROP | `[-]` | Live Snowflake loading remains deferred; `SnowflakeSource` is a typed source stub. |

---

## `automl/loop_context/`

### `loop_context/__init__.py`
| Symbol | Kind | New home | Status |
|---|---|---|---|
| `experiment_id` | function | DROP | `[-]` | Active experiment id comes from `Session.active_experiment_id`. |

### `loop_context/proposer_packet.py`
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `find_prior_experiment` | function | `agent/proposer_context.py` (sub-spec 11) — composes `mlflow.project.list_experiments` + per-id `read_overview` | `[x]` | Cold-start prior-experiment discovery; orders by `created_at` instead of lexicographic experiment id. |
| `gather_proposal_context` | function | `agent/proposer_context.py` | `[x]` | Ported as `gather_proposer_context`; returns the Phase 5 packet shape and drops legacy `top_trials`/learnings/artifact-error keys. |

### `loop_context/queries.py`
| Symbol | Kind | New home | Status |
|---|---|---|---|
| `top_n_by_metric`, `recent_failures`, `strategies_attempted`, `show_trial` | functions | `mlflow/experiment/queries.py`, `experiment/views/queries.py`, `trial/show.py` | `[x]` | Phase 4 landed typed seam rows plus experiment/trial read views |
| `runs_using_strategy`, `runs_in_metric_band` | functions | DROP | `[-]` | No current caller; leaderboard/search views cover the accepted experiment summary gates. |

### `loop_context/summary.py`
| Symbol | Kind | New home | Status |
|---|---|---|---|
| `load_mlflow_context` | function | `experiment/views/summary.py` | `[x]` |
| `build_summary_from_context` | function | `experiment/views/summary.py` | `[x]` |
| `build_summary` | function | `experiment/views/summary.py` | `[x]` |

---

## `automl/mlflow/`

### `mlflow/code_bundle.py`
| Symbol | Kind | New home | Status |
|---|---|---|---|
| `EXCLUDES` | const | DROP | `[-]` |
| `stage_code_bundle` | function | DROP | `[-]` | Code-bundle staging is not part of the accepted Phase 1-7 gates; trial folder `source/model.py` is the preserved source artifact. |

### `mlflow/overview.py`
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `ensure_project_overview` | function | `mlflow/project/overview.py` as `ensure_overview` | `[x]` | Minimal project-overview run support landed for profile artifacts. |
| `write_project_overview_artifacts` | function | `mlflow/project/artifacts.py` | `[x]` | Profile artifact writer/readers cover the accepted project-overview artifact path; broad learning artifact writer remains dropped. |
| `main` (argparse) | function | `cli/project.py` / `cli/experiment.py` / `cli/data.py` | `[x]` | `python -m automl.mlflow.overview` removed; overview/profile entry points are noun CLI verbs. |

### `mlflow/store.py` (1179L — splits across mlflow/{project,experiment,trial,queries}.py)
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `SAFE_NAME_RE` | const | `mlflow/_routing.py` | `[x]` | Implemented as `_SAFE_ROUTE_COMPONENT_RE`. |
| `EXPERIMENT_LEARNING_ARTIFACTS`, `IGNORED_TOP_LEVEL_DATA_ARTIFACTS`, `EXPERIMENT_REFERENCE_ARTIFACTS`, `PROJECT_LEARNING_ARTIFACTS`, `LEARNING_CACHE_ARTIFACTS` | constants | DROP | `[-]` | Project learning subsystem intentionally deferred/dropped from the Phase 1-7 refactor. |
| `MlflowSettings` | dataclass | `mlflow/client.py::Bound` | `[x]` | Replaced by process-bound `Bound`. |
| `SnapshotNameResolution` | dataclass | DROP | `[-]` | Snapshot naming API retired with dataset ids. |
| `project_overview_experiment_name` | function | `mlflow/_routing.py` | `[x]` | Covered by `experiment_route("overview")`. |
| `experiment_name` | function | `mlflow/_routing.py::experiment_route` | `[x]` | |
| `artifact_url`, `run_url` | functions | `mlflow/client.py` | `[x]` | |
| `SNAPSHOT_HASH8_RE`, `SNAPSHOT_NAME_RE` | constants | DROP | `[-]` | Snapshot naming API retired with dataset ids. |
| `resolve_snapshot_name` | function | DROP | `[-]` | Dataset ids are resolved through `data.registry`, not MLflow snapshot-name helpers. |
| `cache_root` | function | DROP | `[-]` | No public MLflow cache-root helper remains. |
| `ensure_project_overview` | function | `mlflow/project/overview.py` as `ensure_overview` | `[x]` | Minimal project-overview run support landed for profile artifacts. |
| `ensure_experiment_overview` | function | `mlflow/experiment/lifecycle.py` as `ensure_overview` | `[x]` | Landed in Phase 4 with typed `ExperimentOverview` |
| `get_trial_summaries` | function | `mlflow/experiment/queries.py` as `list_trials` / `top_n_by_metric` | `[x]` | Landed in Phase 4 with typed `TrialSummary` rows |
| `snapshot_prefix` | function | DROP | `[-]` | Snapshot-specific route helper retired; dataset/eval artifacts use typed URIs and `_routing.bucket_uri_for`. |
| `sync_active_data_snapshot_tags`, `set_active_data_snapshot`, `read_snapshot_index`, `get_active_data_snapshot` | functions | `mlflow/experiment/lifecycle.py` + `data.registry` | `[x]` | Active dataset is experiment tag state; dataset index is data-owned. |
| `get_context` | function | `agent/proposer_context.py` | `[x]` | Replaced by proposer context assembly over typed seam reads. |
| `write_cache_json`, `learning_feature_payloads`, `write_learning_cache` | functions | DROP | `[-]` | Project learning subsystem intentionally deferred/dropped from the Phase 1-7 refactor. |

### `mlflow/tags.py`
| Symbol | Kind | New home | Status |
|---|---|---|---|
| `CREATED_BY_TAG`, `mlflow_created_by`, `set_created_by_tag` | const+functions | `mlflow/tags.py` / lifecycle helpers | `[x]` | `CREATED_BY` tag constant and lifecycle tag write landed; helper wrappers dropped. |

### `mlflow/artifacts/data.py`
| Symbol | Kind | New home | Status |
|---|---|---|---|
| `write_data_contract` → `write_trial_data_contract` | function | `mlflow/trial/artifacts/data.py` | `[x]` |

### `mlflow/artifacts/eval.py`
| Symbol | Kind | New home | Status |
|---|---|---|---|
| `validate_eval_label` | function | `mlflow/trial/artifacts/eval.py` | `[x]` |
| `write_evaluation_results` | function | `mlflow/trial/artifacts/eval.py` (`write_eval`, plus `load_eval`/`list_eval`/`write_eval_index`/`load_eval_index`) | `[x]` |

### `mlflow/artifacts/failure.py`
| Symbol | Kind | New home | Status |
|---|---|---|---|
| `write_error_log` | function | DROP | `[-]` | Runner returns typed failure results; no accepted gate requires persisted error-log artifacts. |

### `mlflow/artifacts/features.py`
| Symbol | Kind | New home | Status |
|---|---|---|---|
| `write_feature_importance`, `write_snapshot_feature_registry`, `write_model_feature_registry` | functions | DROP | `[-]` | Not part of accepted Phase 1-7 behavior; feature registry lineage is covered by data/model contracts. |

### `mlflow/artifacts/gcs_paths.py`
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `SAFE_ROUTE_COMPONENT_RE` | const | `mlflow/_routing.py` | `[x]` | Implemented as `_SAFE_ROUTE_COMPONENT_RE`. |
| `experiment_route`, `route_prefix_for`, `bucket_uri_for` | functions | `mlflow/_routing.py` | `[x]` | Mode/namespace are bound-session fields; `bucket_uri_for(kind="agent_events", run_id=...)` is used by `agent.timeline`. No dynamic hook import remains. |

### `mlflow/artifacts/manifest.py`
| Symbol | Kind | New home | Status |
|---|---|---|---|
| `write_manifest` | function | `mlflow/trial/artifacts/manifest.py` | `[x]` |

### `mlflow/artifacts/model.py`
| Symbol | Kind | New home | Status |
|---|---|---|---|
| `write_model_report` | function | `mlflow/trial/artifacts/model.py` | `[x]` | Model binary and source artifacts landed; legacy report object is dropped. |

### `mlflow/artifacts/predictions.py`
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `PredictionsArtifact`, `write_predictions_gcs` | dataclass+func | `eval/results.py::Predictions` + `mlflow/trial/artifacts/predictions.py` | `[x]` | Includes `write_predictions`, `load_predictions`, and `list_predictions`. |
| `gcs_blob_exists`, `gcs_delete_blob_if_exists`, `gcs_promote_blob`, `get_json_from_gcs`, `read_parquet_from_gcs`, `write_df_to_gcs_as_parquet`, `write_json_to_gcs` | shim funcs | DROP — use `utils/io/gcs.py` | `[-]` | Duplicated shims |

### `mlflow/artifacts/start_trial.py`
| Symbol | Kind | New home | Status |
|---|---|---|---|
| `RequiredTags` | NamedTuple | DROP | `[-]` | Replaced by canonical tag constants and `mlflow.trial.start`. |
| `start_trial` | function | `mlflow/trial.py` | `[x]` | Landed as `mlflow.trial.start` / `active`. |

### `mlflow/artifacts/timing.py`
| Symbol | Kind | New home | Status |
|---|---|---|---|
| `write_timing`, `headline_metrics` | functions | DROP | `[-]` | Timing report is typed in `trial.metadata`; no accepted gate requires a standalone writer. |

### `mlflow/artifacts/validation.py`
| Symbol | Kind | New home | Status |
|---|---|---|---|
| `validation_status`, `write_validation_report` | functions | DROP | `[-]` | Validation reports are returned through `validate` and not persisted as MLflow artifacts in accepted gates. |

---

## `automl/profile/`

### `profile/core.py`
| Symbol | Kind | New home | Status |
|---|---|---|---|
| `generate_deterministic_charts`, `distill_observations`, `build_data_card`, `build_snapshot_data_card` | functions | `data/profile.py` | `[x]` | Implemented as the profile checks/chart summaries used by `Profile`. |
| `write_profile_outputs`, `write_snapshot_profile_outputs` | functions | `data/profile.py` + `mlflow/project/artifacts.py` | `[x]` | Profile outputs are written through the MLflow project artifact seam. |
| `validate_output_path` | function | DROP | `[-]` | Local output-path API removed; artifacts persist through the seam. |
| `main` | argparse | `cli/data.py` | `[x]` | `data profile` owns the CLI entry. |

### `profile/snapshot.py`
| Symbol | Kind | New home | Status |
|---|---|---|---|
| `ProfileResult` | dataclass | `data/profile.py` | `[x]` |
| `profile_active_snapshot`, `profile_snapshot` | functions | `data/profile.py` | `[x]` |
| `main` | argparse | `cli/data.py` | `[x]` | `data profile` owns the CLI entry. |

---

## `automl/propose/`

### `propose/__init__.py`
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `SLUG_RE` | const | `utils/` | `[x]` | Implemented in `utils/slug.py` and consumed by `agent.proposal`/`agent.checks`; future `trial.create` can reuse it without a `trial → agent` cycle. |
| `validate` | function | DROP — logic absorbed into `agent/checks.py::proposal_schema`; `validate.proposal` orchestrator calls it directly | `[-]` | Sub-spec 04 Q5; 3 callers updated (`cli/propose.py`, `cli/trial.py:76`, `validate/builtin/proposal_checks.py`) |

### `propose/schema.py`
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `REQUIRED_FIELDS`, `OPTIONAL_FIELDS`, `DISALLOWED_FIELDS` | constants | `agent/proposal.py` | `[x]` | Replaced by the `Proposal` dataclass field roster plus `DISALLOWED = ("parent_id",)`. |
| `Issue` (proposal-specific) | dataclass | DROP — unify into `validate/base.py::Issue` | `[-]` | Sub-spec 04 Q5; canonical type returned directly, no translation |
| `ValidationReport` (proposal-specific) | dataclass | DROP — unify into `validate/base.py::ValidationReport` | `[-]` | Same |

---

## `automl/runner/`

### `runner/_execute.py` (1496L) — sub-spec 08 (Tier-1 cohesive modules, NO `stages/`)
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `TrialResult` | dataclass | `runner/trial.py` | `[x]` | Phase 1 thin result landed. |
| `run_trial` | function | `runner/trial.py` (straight-line chain) | `[x]` | 08: data_load is fit-slice-only; builds `TrialDataContract` via `mlflow.trial.artifacts.write_trial_data_contract`; eval via `evaluate()→EvalResult`; pre-fit gate on fit frame |
| `main` | argparse | `cli/trial.py` | `[x]` | Phase 6 `trial run` delegates to `runner.run_trial`. |
| (failure `finally` — both paths) | logic | `runner/trial.py` | `[-]` | Failure-manifest spine was dropped; runner returns typed `TrialResult(FAILED)` and successful paths write typed artifacts. |

### `runner/_stages.py` (1307L — dissolved into cohesive modules, NOT one file) — sub-spec 08
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `WARMUP_ITERATIONS`, `MEASURED_ITERATIONS`, `REPEAT_GROUPS`, `TRIM_FRACTION` | constants | DROP | `[-]` | Subprocess pyfunc benchmark did not survive the accepted runner scope. |
| validation fixture + round-trip + GCS publish | functions | `runner/contract.py` / `validate.targets` | `[x]` | Replaced by fit-frame validation, post-fit contract checks, model persistence, and eval gates. |
| manifest / artifact-listing / failure-manifest assembly | functions | DROP | `[-]` | Slim manifest spine dropped; typed artifacts and `trial.show` provide artifact listing. |
| collision-safe model import + code-bundle staging | functions | `runner/trial.py` | `[x]` | Trial-folder `model.py` import is collision-safe; code-bundle staging replaced by `source/model.py`. |
| post-fit registry/method-contract checks | functions | `runner/contract.py` | `[x]` | Phase 1 post-fit attrs, registry preservation, target-set, and BaseModel method ownership checks landed. |
| (trial folder path + universe-isolation verify) | new | `runner/paths.py` | `[x]` | 08 Q1 |
| `_write_error_log` (shim), `_load_trial_model_module`, `_frame_shape` | functions | DROP — dead code | `[-]` | 08 Q6; sweep `tests/` for indirect callers at impl |
| `_hash_seed`, `_resolve_git_commit`, `_safe_error_tag`, `_TimingRecorder` | functions | DROP | `[-]` | Not needed by the accepted straight-line runner path. |
| `from automl.runner._stages import *` | import | replace with explicit imports | `[x]` | New runner imports directly from the new domains; no `_stages` star import. |

### `runner/template.py`
| Symbol | Kind | New home | Status |
|---|---|---|---|
| (template strings) | const | `runner/template.py` | `[x]` |

---

## `automl/session/`

### `session/lock.py`
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `DEFAULT_STALE_AFTER_SECONDS`, `LOCK_ERROR` | constants | `runner/session_lock.py` | `[x]` | |
| `lock_dir`, `lock_path` | functions | `runner/session_lock.py` | `[x]` | |
| `is_locked`, `acquire`, `release`, `session_lock` (cm) | functions | `runner/session_lock.py` | `[x]` | Phase 6 post-implementation review found and fixed the missing real lock behavior. |
| `main` | argparse | `cli/trial.py` (`automl trial lock {acquire,release}` → `runner.session_lock`, 00 line 456) | `[x]` | `python -m` removed; CLI delegates to runner session-lock APIs. |

---

## `automl/trial/` (most ops move to trial/)

### `trial/cleanup.py`
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `cleanup(project, project_root, trial_id, run_id, dry_run, confirm_project)` | function | `trial/cleanup.py::delete(run_id, *, apply, hard_delete, session)` (thin wrapper → `project/cleanup.py` cascade engine) | `[x]` | Resolved by sub-spec 03 §9. Dropped args: `trial_id` (slug selector — run_id is canonical per sub-spec 02 §6.3.1); `dry_run` (read from session); `confirm_project` (no interactive confirmation in new design) |

### `trial/creation.py`
| Symbol | Kind | New home | Status |
|---|---|---|---|
| `SLUG_RE` | const | `utils/` (dedupe with propose/) | `[x]` | Sub-spec 10: `utils/`, not `agent/` — avoids `trial → agent` cycle |
| `create` | function | `trial/create.py` | `[x]` | Phase 7 builds the trial folder at the **mode-segregated** path via `runner`'s path helper |
| `_next_trial_number_from_mlflow` (+ `_run_trial_number`) | function | `mlflow/experiment/` as typed read `next_trial_number(...)` | `[x]` | Phase 1 seam read landed; runner uses it for exec-time assignment. Later `trial.create` can reuse the same seam API. |

### `trial/fork.py`
| Symbol | Kind | New home | Status |
|---|---|---|---|
| `fork` | function | `trial/fork.py` | `[x]` |

### `trial/packaging.py`
| Symbol | Kind | New home | Status |
|---|---|---|---|
| `package_model` | function | `trial/packaging.py` (sub-spec 10) | `[x]` | **Sub-spec 06 correction:** notebook-class → `model.py` *source extraction* (trial-authoring), not serialization. Was wrongly mapped to `model/packaging.py` |

### `trial/promotion.py`
| Symbol | Kind | New home | Status |
|---|---|---|---|
| `promote` | function | `trial/promote.py` | `[x]` |

### `trial/run.py`
| Symbol | Kind | New home | Status |
|---|---|---|---|
| `run` (1-line delegate) | function | DROP — caller goes direct to `runner.run_trial` | `[-]` |

---

## `automl/utils/`

### `utils/logging.py`
| Symbol | Kind | New home | Status |
|---|---|---|---|
| `configure_logging` | function | `utils/logging.py` (kept, actually used) | `[x]` | Phase 1 leaf helper landed with unit coverage. |

---

## `automl/validate/`

### `validate/builtin/__init__.py`
| Symbol | Kind | New home | Status |
|---|---|---|---|
| `BUILTIN_MODULES` | const | DROP — replaced by per-domain `checks.py` registration | `[-]` |
| `register_all` | function | DROP — registry imports from each domain's `checks.py` | `[-]` |

### `validate/builtin/config_checks.py`
| Symbol | Kind | New home | Status |
|---|---|---|---|
| `check_run_config` | check | `validate.targets.project` / `project.run_config` constructors | `[x]` |

### `validate/builtin/contract_checks.py`
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `check_task_module_exports` | check | `validate.targets.project` / `ProjectConfig.load` | `[x]` | |
| `check_data_module_exports` | check | `validate.targets.project` / `ProjectConfig.load` | `[x]` | |
| `check_evaluation_module_exports` | check | `validate.targets.project` / `ProjectConfig.load` | `[x]` | |
| `_probe_evaluation_shape` | helper | DROP | `[-]` | Eval spec shape validates through `EvalSpec` itself. |
| `_load_project_module` | helper | `project/_import.py` | `[x]` | |
| `check_project_placeholders` | check | DROP — replaced by `None`-semantics (sub-spec 01 §5) | `[-]` | |

### `validate/builtin/env_checks.py`
| Symbol | Kind | New home | Status |
|---|---|---|---|
| `check_gcs_bucket`, `check_gcs_prefix`, `check_mlflow_tracking_uri` | checks | `ProjectConfig.load` / seam binding warnings | `[x]` |

### `validate/builtin/model_checks.py`
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `REQUIRED_POST_FIT_ATTRS` | const | `model/checks.py` | `[x]` | |
| `ModelProbe` | dataclass | DROP — sub-spec 04 Q4 | `[-]` | Replaced by private `_try_fit(cls, df, registry, *, seed=0) -> (instance, error, error_stage)` inside `validate/targets.py` |
| `make_model_probe`, `sample_load_failed_probe` | functions | DROP — sub-spec 04 Q4 | `[-]` | Logic absorbed into the orchestrator's private `_try_fit` |
| `check_subclass_basemodel`, `check_fit_succeeds`, `check_post_fit_attrs_set` | checks | `model/checks.py` (signature changed: take `cls, instance, error, error_stage` as kwargs instead of `ModelProbe`) | `[x]` | Drops the `sample_kind` arg too. Sub-spec 06 adds sibling `check_required_transformers` here |

### `validate/builtin/proposal_checks.py`
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `check_proposal` (adapter) | check | DROP — absorbed into `agent/checks.py::proposal_schema` | `[-]` | Sub-spec 04 Q5 + 11 §4: `proposal_schema(proposal: dict, *, session=None) -> list[Issue]` returns canonical `validate.Issue`; session-resolves the allow-list (drops `allowed_dependencies` param + CLI flags; fixes the `cli/trial.py` tautology) |

### `validate/registry.py`
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `register`, `get_checks` | functions | DROP — sub-spec 04 Q1 deletes the decorator + `_CHECKS` global | `[-]` | Built-in checks become direct function calls |
| `discover_project_checks` | function | DROP | `[-]` | Project-local validator discovery was dropped; built-in orchestrators are direct. |
| `_CHECKS`, `_DISCOVERED_PROJECTS` | module-level state | DROP | `[-]` | Mutable globals gone with the decorator |
| `_RESET_FOR_TESTS` | helper | DROP | `[-]` | No mutable registry cache remains. |
| (whole `registry.py` file) | — | DROP | `[-]` | File deleted; surviving pieces relocated above |

### `validate/synthetic.py`
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `make_synthetic_fixture` | function | `validate/synthetic.py` (kept; signature simplified to `(*, rows: int = 50)`) | `[x]` | Legacy `n_numeric`/`n_categorical`/`target_col`/`seed` kwargs dropped; always defaults |

### `validate/targets.py`
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `REAL_SAMPLE_ROWS = 200` | const | DROP from validate; runner owns it now | `[-]` | Q4 A1 — runner takes `.head(200)` itself |
| `model`, `proposal`, `project` | orchestrators | `validate/targets.py` | `[x]` | Phase 6 adds bounded structural `project` validation; broader per-domain check rows above remain individually tracked. |
| `config`, `contracts` | orchestrators | DROP — sub-spec 04 Q3 collapses into `project` | `[-]` | Checks still run; only the separate orchestrators + CLI sub-verbs are removed |
| `_model_real_sample_probe`, `_real_sample_project_context`, `_coerce_project_context`, `_run`, `_call_check` | private helpers | DROP — sub-spec 04 Q1+Q4 obviate them | `[-]` | No `inspect.signature` filtering; no internal data loading |

### `validate/types.py`
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `Issue`, `ValidationReport` | dataclasses | `validate/base.py` (renamed) | `[x]` | Canonical report now has JSON helpers, `schema_version`, and issue `location`. |
| `Severity` | Literal | `validate/base.py` as `Literal["error", "warning"]` | `[x]` | Q6b drops `"info"`. |
| `Target` | Literal | `validate/base.py` as `Literal["project", "model", "proposal"]` | `[x]` | Q3 prunes from 5 to 3. |
| `CheckSpec` | dataclass | DROP — sub-spec 04 Q1 | `[-]` | No callers post-refactor |
| `ValidationReport.raise_if_failed` | method | DROP — sub-spec 04 Q6c | `[-]` | No caller |

### `automl/errors.py` (sub-spec 04 carry)
| Symbol | Kind | New home | Status | Notes |
|---|---|---|---|---|
| `ValidationError` | exception | DROP — sub-spec 04 Q6c | `[-]` | Paired with `raise_if_failed` removal |

---

# Outside the library (orientation only — not in Appendix A)

## `skills/`
| Item | New home | Status | Notes |
|---|---|---|---|
| `skills/automl/SKILL.md` + `scripts/` | `skills/automl/` (kept; prose updated for new CLI verbs) | `[x]` | Phase 6 updates render/preflight scripts and command docs. |
| `skills/automl-guide/SKILL.md` | `skills/automl-guide/` (kept; prose updated) | `[x]` | |
| `skills/coder/SKILL.md` | `skills/coder/` (kept) | `[x]` | |
| `skills/inspect/SKILL.md` + `scripts/` | `skills/inspect/` (kept) | `[x]` | Uses noun-first experiment/trial inspection commands. |
| `skills/profile/SKILL.md` + `scripts/` | `skills/profile/` (kept) | `[x]` | Uses `data profile`. |
| `skills/propose/SKILL.md` + `scripts/` | `skills/propose/` (kept; proposal rename deferred/dropped) | `[x]` | Kept folder name for existing skill packaging; commands now use `experiment proposer-context` and `validate proposal`. |
| `skills/setup/SKILL.md` | `skills/setup/` (kept) | `[x]` | Uses `project init` / `validate project`. |
| `skills/validate/SKILL.md` | `skills/validate/` (kept) | `[x]` | Uses `validate project|model|proposal`. |

## `agents/`
| Item | New home | Status | Notes |
|---|---|---|---|
| `agents/automl-proposer.md` | `agents/automl-proposer.md` (kept; prose updated for new shapes) | `[x]` | Input roster updated to the Phase 5 proposer packet. |
| `agents/automl-coder.md` | `agents/automl-coder.md` (kept; prose updated) | `[x]` | Phase 6 command/import prose updated. |

## `hooks/`
| Item | New home | Status | Notes |
|---|---|---|---|
| `hooks/hooks.json` | `hooks/hooks.json` (kept; possibly retargeted) | `[x]` | Existing command now targets the thin stub; launcher env supplies project/session context. |
| `hooks/agent_timeline.py` (1955L) | THIN STUB → `agent.timeline.handle_event()` | `[x]` | Replaced by an argparse/stdin stub delegating to `agent.timeline.handle_event()`/`publish()`. |

## `references/`
| Item | New home | Status |
|---|---|---|
| `references/setup/*`, `references/implement/*`, `references/loop/*` | `references/` (kept; prose updated) | `[x]` |

## `projects/`
| Item | New home | Status |
|---|---|---|
| `projects/payment_routing/`, `projects/example_homecredit/` | `projects/` (kept; per-project SQL queries + PROJECT_INSTRUCTIONS.md preserved) | `[x]` |
| Project-local `config.py` template (uses legacy import paths) | Update import paths to new modules | `[x]` | Phase 7 updates `payment_routing`; `example_homecredit` was already on new imports. |
| Project-local `data/pipeline.py` overrides | Update subclass imports | `[-]` | No project-local pipeline override files remain in the fresh tree. |

## Top-level files
| Item | New home | Status |
|---|---|---|
| `CHANGELOG.md`, `README.md`, `CLAUDE.md` | kept; prose updated | `[x]` | Refactor status prose lives in README/plan docs; absent optional files require no migration. |
| `.env.example`, `.gitignore`, `pyproject.toml`, `uv.lock`, `pyrightconfig.json`, `.vscode/` | kept; minor updates | `[x]` | `pyproject.toml` testpaths/package find updated for cutover; absent optional files require no migration. |

---

# Audit notes — closed concerns

- **Three parallel `Issue`/`ValidationReport` dataclasses** resolved in sub-spec 04 Q5 and
  Phase 5. `validate/base.py` is canonical; proposal validation uses `validate.proposal`, and
  `propose.validate()` was dropped.
- **Duplicated GCS shim functions** collapsed into `utils/io/gcs.py`; MLflow artifact modules
  and data/profile code use the utility seam.
- **MLflow artifact taxonomy constants** were resolved by the seam split: canonical tag keys live
  in `mlflow/tags.py`; noun-specific artifact I/O lives under `mlflow/project/artifacts.py`,
  `mlflow/trial/artifacts/`, and eval artifact helpers. Legacy learning-cache constants were
  intentionally dropped per sub-spec 02.
- **`SLUG_RE` duplication** resolved to the neutral `utils/slug.py`, imported by `agent/` and
  `trial/`.
- **Snapshot regex duplication** retired with the snapshot-to-dataset rename; dataset identifiers
  are validated in `data/dataset.py` and route/path slugs use runner-local sanitization.
- **`validate_run_data_contract` duplication** resolved by the new `data/contract.py`
  `TrialDataContract` + `validate_trial_data_contract` shape.
- **Library `__main__` blocks** were removed from library modules. The only `automl/` entry point
  is `cli/__main__.py`; user-facing execution goes through noun-first `uv run automl ...` verbs.

---

# How to use this checklist

1. **During each sub-spec** — when we settle a domain's interface, update that domain's rows (mostly Status column).
2. **During implementation** — flip Status to `[/]` when partial; `[x]` when complete.
3. **At audit time (before cutover)** — every row must be `[x]` or `[-]`. Anything still `[ ]`, `[/]`, or `[?]` is unfinished business.

Run `grep "\[?\]"` to find anything still uncertain. Run `grep "\[/\]"` to find partial.
