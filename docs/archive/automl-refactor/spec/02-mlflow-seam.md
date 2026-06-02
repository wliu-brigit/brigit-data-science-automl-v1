# MLflow Seam Interfaces — Sub-Spec Design

**Date:** 2026-05-22
**Parent spec:** `00-structural-design.md` §9.1, §15.1 Priority 2
**Status:** Design approved; ready to inform implementation.
**Scope:** The public API, internal organization, and operational invariants of `automl/mlflow/` — the framework's persistence layer.

This is the seam every domain depends on. The structural spec settled *that* `mlflow/` exists and that domain code never `import mlflow` directly; this sub-spec settles *what's in it* — function signatures, return types, error handling, and discoverability.

---

## 1. Why this sub-spec exists

Today's `automl/mlflow/store.py` is 1179 lines and mixes routing, queries, project-overview, experiment-overview, trial summaries, snapshot management, and learning-cache writes. Domain code (`runner/`, `experiment/views/`, etc.) imports from it ad-hoc and uses MLflow's API directly through it. The two consequences are exactly the problems the refactor exists to fix:

- **Discoverability is poor.** A new DS has no way to know "what's queryable from MLflow at the project level" or "what artifacts a trial has" without reading the source.
- **The seam is leaky.** Domain code knows about MLflow's quirks (experiment naming rules, tag conventions, search filter syntax). When MLflow changes — or when we want to mock it for tests — the blast radius is the whole library.

This sub-spec defines a `mlflow/` package whose public API is the *only* surface domain code touches, organized so that what's available at each level (project / experiment / trial) is discoverable by tab-completion and documented in one place.

---

## 2. What we have today

Quick inventory of the current `automl/mlflow/`:

| File | Lines | Role |
|---|---|---|
| `mlflow/store.py` | 1179 | Catch-all: routing string construction, project + experiment overview, trial summaries, active dataset tags, snapshot resolution, learning cache |
| `mlflow/overview.py` | 80 | Thin wrappers over `store.py`'s project-overview helpers; also has a `__main__` block |
| `mlflow/code_bundle.py` | 80 | `stage_code_bundle` — stages a tar artifact at trial start |
| `mlflow/tags.py` | 30 | One constant (`CREATED_BY_TAG`) + two helpers |
| `mlflow/artifacts/start_trial.py` | — | `start_trial` (lives under artifacts/ for historical reasons) |
| `mlflow/artifacts/data.py` | — | `write_data_contract` (legacy; renamed to `write_trial_data_contract`) |
| `mlflow/artifacts/eval.py` | — | `write_evaluation_results`, `validate_eval_label` |
| `mlflow/artifacts/failure.py` | — | `write_error_log` |
| `mlflow/artifacts/features.py` | — | `write_feature_importance` + registry writers |
| `mlflow/artifacts/gcs_paths.py` | — | `experiment_route`, `route_prefix_for`, `bucket_uri_for` (path construction) |
| `mlflow/artifacts/manifest.py` | — | `write_manifest` |
| `mlflow/artifacts/model.py` | — | `write_model_report` |
| `mlflow/artifacts/predictions.py` | 408 | `write_predictions_gcs` + manual partial-rollback + duplicated GCS shim functions |
| `mlflow/artifacts/timing.py` | — | `write_timing`, `headline_metrics` |
| `mlflow/artifacts/validation.py` | — | `write_validation_report`, `validation_status` |

Cross-cutting issues:
- Routing logic split between `store.py` and `artifacts/gcs_paths.py` (two near-identical path constructors)
- `dry_run: bool` vs `run_mode: Literal["dry_run", "full_run"]` representations both used
- Tag keys mostly hardcoded as string literals throughout the codebase rather than centralized constants
- Per-artifact partial-rollback logic duplicated (predictions.py has 30 lines of try/except for what other writers don't even attempt)
- GCS helper functions defined as shims inside `data/pipeline.py` AND inside `mlflow/artifacts/predictions.py` — two parallel implementations

---

## 3. Design pillars

Seven invariants that fall out of the sub-spec discussion. The rest of this document derives from these.

### 3.1 Connection state is bound once, not passed everywhere

Process-level connection state (`tracking_uri`, `bucket`, `gcs_prefix`, `project_name`, `experiment_id`, `dry_run`, `namespace`) is set once by `mlflow.bind(...)` and cached in a contextvar inside `mlflow/client.py`. Public functions do NOT take these as parameters.

`bind()` is most commonly called by `automl.use_project()` as part of session bootstrap — but it can also be called directly by **non-session callers**: admin scripts, hook subprocesses spawned by Claude Code, migration tools, or tests. Anything that touches MLflow must call `bind()` first; calling `automl.use_project()` is one way to do that, not the only way. (See §5 for details.)

**The legacy `run_mode` routing string is collapsed (10 §7.2 cross-cutting + 11 #5).** The two-valued string `"dry_run"`/`"full_run"` that legacy code threaded through `gcs_paths.py`, MLflow experiment names, and data/eval routing is **gone**: mode lives once on `bound().dry_run` (a `bool`, fed from `session.dry_run`), and the universe token exists only as a **conditional `dry_run/` prefix inside `_routing.py`**. No public function takes a `run_mode`/`dry_run` parameter; no `"full_run"` literal is ever stored. **`namespace` (renamed from legacy `route_namespace`) is a *separate, surviving* isolation dimension** from the mode — an arbitrary prefix (e.g. `"qa"`) for a **full-universe** isolated sandbox (segregates MLflow experiment names + GCS + local trial dirs), **orthogonal to and composable with** dry_run (route segment order: `[<namespace>/][dry_run/]<project>/<experiment_id>`). Unlike legacy (where it defaulted `""` and was never wired — hence dead), it is now fed by the top-level `--namespace` flag → `Session.namespace` (sub-spec 01). Its purpose: full-fidelity QA/test runs the user can clean up without touching the real namespace — distinct from dry_run, which is a *reduced-fidelity* (data-subset) smoke universe. The "kill route_namespace" note in sub-spec 11 referred only to the agent timeline's dead `""` usage + route-*string* parsing, not this bound field.

### 3.2 Identifiers are level-specific, not universal

- **Project level**: no identifier (one project per session)
- **Experiment level**: `experiment_id: str | None = None` — defaults to the bound experiment_id
- **Trial level**: `run_id: str` — always required (no "active trial" outside the runner's context manager)

There is **no `Route` object**. Each function takes exactly the identifier needed at its level.

### 3.3 Per-noun folders mirror the hierarchy

`mlflow/project/`, `mlflow/experiment/`, `mlflow/trial/` are folders, not files. Internal modules group related operations (lifecycle / queries / snapshots / artifacts). `__init__.py` re-exports the public surface so import paths stay shallow.

### 3.4 One unified logging API for active + post-hoc

`mlflow.trial.log_metric(run_id, ...)` works whether `run_id` came from `mlflow.trial.active(...)` (runner context) or from an existing trial (post-hoc hook). No `append_*` / `log_*` split.

### 3.5 GCS-then-MLflow ordering is the writer contract

Every typed artifact writer in `mlflow/<noun>/artifacts/`:
1. Writes payload bytes to GCS first (multi-step writes use the `_atomic.py` partial-rollback helper)
2. Logs the GCS URI to MLflow as the commit point
3. Raises `StorageError` on failure of either step

Orphan GCS blobs (the MLflow-fails-after-GCS-succeeds case) are recoverable via the cleanup verb. MLflow is the source-of-truth ledger.

### 3.6 Schemas are additive-only with a `schema_version` placeholder

Every typed artifact has `schema_version: int = 1` from day one. New fields are added with `Optional[...] = None` defaults; nothing is renamed or removed. **Loaders use a `from_dict` classmethod that strips unknown keys** so an older reader can deserialize a newer-version payload without crashing (additive-only is only forward-compatible if the deserializer ignores extras — bare `Cls(**payload)` raises `TypeError` on unknown keys). Version-dispatch code is written only when we actually break a schema. See §8.

### 3.7 Two-tier artifact strategy: typed + loose

- **Typed tier**: formal writer/loader in `mlflow/<noun>/artifacts/*.py` with a dataclass payload schema. Used for stable artifacts (eval results, data contract, predictions, model, features, timing, validation, manifest). Goes through the GCS-then-MLflow contract.
- **Loose tier**: `mlflow.trial.log_json(run_id, name, payload: dict)` — no schema, no validation. Used for experimental data, debug payloads, agent reports (where the schema is in flux or doesn't pay rent yet). Promotion from loose to typed is a small refactor when shape stabilizes.

---

## 4. Folder layout

```
automl/mlflow/
├── __init__.py
├── client.py                ← bind(), bound() — connection state in contextvar; run_url(), artifact_url() (§6.4)
├── tags.py                  ← canonical tag-key constants
├── _atomic.py               ← shared partial-write helper (private)
├── _routing.py              ← internal path construction (uses bound state): <project>/<experiment_id> rules,
│                              conditional dry_run/ prefix (from bound().dry_run — no run_mode/"full_run" string),
│                              and the deterministic agent-events GCS prefix from (session, run_id) — sub-spec 11 #6,
│                              the single source both the runner (writes) and the timeline (uploads) call, replacing
│                              the legacy runner→timeline manifest handshake
│
├── project/
│   ├── __init__.py          ← re-exports: read_overview, write_overview, ensure_overview, list_experiments
│   ├── overview.py          ← project-overview run read/write
│   └── artifacts.py         ← dataset_index, data_card, data_learning,
│                              dataset manifest (write_dataset / read_dataset),
│                              dataset-id resolution (resolve_dataset_id, assign_dataset_id),
│                              profile (write_profile / read_profile)
│                              — all project-scoped artifacts. See §6.1 + sub-spec 05.
│
├── experiment/
│   ├── __init__.py          ← re-exports
│   ├── lifecycle.py         ← ensure(), ensure_overview, read/write_overview
│   ├── queries.py           ← list_trials, top_n_by_metric, search_runs
│   ├── datasets.py          ← list_datasets_for_experiment, get/set_active_dataset, resolve_dataset_id
│   │                          (was snapshots.py — renamed per sub-spec 05 Q1 vocab retirement)
│   └── artifacts.py         ← leaderboard_snapshot, session reports
│                              (no diagnostics.py — zero placeholder files, sub-spec 09 §Q4)
│
└── trial/
    ├── __init__.py          ← re-exports the entire trial surface
    ├── lifecycle.py         ← active() (cm), start(), end()
    ├── logging.py           ← log_metric(s), log_param(s), set_tag(s), log_json
    ├── reads.py             ← get_details, get_metrics, list_artifacts
    └── artifacts/           ← sub-folder because volume warrants it
        ├── __init__.py      ← FLAT re-exports — see §6.3.4
        ├── eval.py          ← write_eval, load_eval, list_eval (MULTI-INSTANCE)
        ├── predictions.py   ← write_predictions, load_predictions, list_predictions (MULTI-INSTANCE)
        ├── data.py          ← write_trial_data_contract, load_trial_data_contract
        ├── model.py         ← write_model, load_model
        ├── model_report.py  ← write_model_report, load_model_report
        ├── features.py      ← write_feature_importance, write_feature_registry, loaders
        ├── timing.py        ← write_timing, load_timing
        ├── validation.py    ← write_validation_report, load_validation_report
        ├── failure.py       ← write_failure
        ├── manifest.py      ← write_manifest, load_manifest
        └── code_bundle.py   ← stage_code_bundle, fetch_code_bundle
```

**Why this shape:**
- `project/` and `experiment/` are folders for symmetry, even though they're smaller than `trial/`. Symmetry beats minimizing folder count.
- `trial/artifacts/` is its own sub-folder because there are 10+ writers and bundling them into one file (`trial/artifacts.py`) would be ~1500 lines.
- `_atomic.py` and `_routing.py` are private (`_` prefix); they are framework internals, not part of the public surface.
- **No `diagnostics.py` placeholder** (sub-spec 09 §Q4 — zero placeholder files). `recent_failures` / `strategies_attempted` are in-scope **view** helpers (`experiment/views/queries.py`, composing the seam); the no-caller analytics (`runs_using_strategy`, `runs_in_metric_band`) are recorded as deferred in the living docs and get a seam search + view helper when a real caller appears — not an empty file now.

**Import paths the DS sees:**

```python
mlflow.project.read_overview()
mlflow.experiment.list_trials()
mlflow.experiment.top_n_by_metric("auc")
mlflow.trial.log_metric(run_id, "auc", 0.85)
mlflow.trial.artifacts.write_eval(run_id, label="train", payload=...)
mlflow.trial.artifacts.load_eval(run_id, label="train")
```

Never deeper than three levels in normal usage. Sub-folders under `artifacts/` are flattened by `__init__.py` re-exports.

---

## 5. Connection state — `bind()` and `bound()`

`mlflow/client.py` holds process-level connection state in a contextvar. `automl.use_project()` calls `bind()` as part of session bootstrap. Domain code never sees these args.

```python
# mlflow/client.py
from dataclasses import dataclass
from contextvars import ContextVar

@dataclass(frozen=True)
class _Bound:
    tracking_uri: str
    bucket: str
    gcs_prefix: str
    project_name: str
    experiment_id: str | None    # may be None during exploration
    dry_run: bool = False
    namespace: str = ""          # isolation prefix (e.g. "qa"); "" = real. Full-universe segment, orthogonal to dry_run. Renamed from legacy route_namespace.

_BOUND: ContextVar[_Bound | None] = ContextVar("automl_mlflow_bound", default=None)


def bind(
    *,
    tracking_uri: str,
    bucket: str,
    gcs_prefix: str,
    project_name: str,
    experiment_id: str | None = None,
    dry_run: bool = False,
    namespace: str = "",
) -> None:
    """Set process-level connection + routing state. Called by automl.use_project()."""
    _BOUND.set(_Bound(
        tracking_uri=tracking_uri,
        bucket=bucket,
        gcs_prefix=gcs_prefix,
        project_name=project_name,
        experiment_id=experiment_id,
        dry_run=dry_run,
        namespace=namespace,
    ))


def bound() -> _Bound:
    """Internal accessor. Raises StorageError if not bound."""
    b = _BOUND.get()
    if b is None:
        raise StorageError("MLflow not bound; call automl.use_project(...) first")
    return b


def raw() -> "mlflow.tracking.MlflowClient":
    """Low-level escape hatch — return the underlying PyPI MlflowClient.

    Use only when no wrapper exists for what you need. Domain code that
    calls this should be reviewed; if a pattern repeats, lift it into mlflow/."""
    import mlflow
    return mlflow.tracking.MlflowClient(tracking_uri=bound().tracking_uri)
```

**`bind()` is process-level.** It clobbers the contextvar — no Token preservation. Callers come in three shapes:

- **Session callers** (`automl.use_project()`, `automl.update_session()`, `automl.active_session()`) — re-fire `bind()` in lock-step with session changes (see §12). `active_session()` correctly saves and restores via the `Session` contextvar Token, which transitively scopes the bind.
- **Non-session callers** (admin scripts, hook subprocesses, migration tools, tests) — call `bind(...)` directly with their own connection args. No `use_project()` needed.
- **Concurrent / async use** — Use `automl.active_session()`, which is the only async-safe pattern. Calling `bind()` directly from inside async tasks is unsafe (will clobber sibling coroutines' state); reserve direct `bind()` for top-level / synchronous setup.

**`bound()` is internal.** Domain code never calls it directly. The `mlflow/<noun>/...` functions call it to fetch what they need from the bound state.

**`raw()` is the escape hatch.** Returns the PyPI `MlflowClient`. Used when we need an MLflow operation we haven't wrapped yet; the existence of a `raw()` call is a signal that the wrapper surface needs to grow.

---

## 6. Per-noun function reference

### 6.1 `mlflow.project.*`

One project per bound session. No identifier ever needed.

**Overview (`project/overview.py`):**

| Function | Returns | Notes |
|---|---|---|
| `read_overview() -> ProjectOverview \| None` | `ProjectOverview` in `project/overview.py` | None when overview run doesn't exist yet |
| `write_overview(overview: ProjectOverview) -> None` | — | Updates the project-overview run (tags / params replaced atomically) |
| `ensure_overview() -> ProjectOverview` | `ProjectOverview` | Creates project-overview MLflow experiment + run if missing; returns current state |
| `list_experiments() -> list[str]` | list of **logical** `experiment_id` strings | Applies the name-filter rules at the seam: prefix-match `<project>/`, exclude the `overview` experiment, exclude nested names (sub-spec 09 §8.1). Returns logical experiment ids, not raw MLflow experiment names. |

**Project-scoped artifacts (`project/artifacts.py`)** — per sub-spec 05 Q3/Q5, this file hosts Dataset manifest + index + profile writers (project-scoped artifacts; Datasets are project-scoped per §5.1):

| Function | Returns | Notes |
|---|---|---|
| `write_dataset(payload: Dataset, *, data_parquet: bytes, registry_csv: bytes) -> DatasetRef` | `DatasetRef` | Writes manifest.json + data.parquet + feature_registry.csv to GCS at `<bucket>/<project>/data/datasets/<dataset_id>/`. Adds entry to `dataset_index.json` on the project-overview run. GCS-then-MLflow ordering per §3.5. |
| `read_dataset(dataset_id: str) -> Dataset` | `Dataset` | Loads manifest.json from GCS; returns typed Dataset (Dataset.component_hashes still need separate cross-check via L2 validator) |
| `read_dataset_index() -> DatasetIndex` | `DatasetIndex` | Reads `dataset_index.json` from the project-overview run. The `active_dataset_id` field is populated by the data domain's `list_datasets()` from the *experiment*-overview tag, not from this artifact (the index is project-scoped, the active pin is per-experiment). |
| `resolve_dataset_id(identity_hash: str) -> tuple[str, bool]` | `(dataset_id, was_new)` | Project-scoped resolution: if a Dataset with this identity_hash exists in `dataset_index.json`, return its existing `dataset_id` + `False`; else mint a new `v<n+1>_<hash8>` id (incrementing the highest version counter) + `True`. Replaces today's `mlflow/store.py:resolve_snapshot_name` + `SnapshotNameResolution` dataclass. |
| `write_profile(dataset_id: str, *, local_dir: Path) -> dict[str, str]` | URIs dict | Uploads profile artifacts (data_card.json, charts/*.png, data_observations.json, profile_manifest.json) under `runs:/<project_overview_run>/<dataset_id>/profile/`. Returns the URI dict the `Profile` typed object embeds. Per sub-spec 05 Q5 bug-fix: lands under project-overview run, NOT experiment-overview run as today. |
| `read_profile(dataset_id: str) -> Profile \| None` | `Profile \| None` | Returns None if no profile exists for this dataset_id. |
| `list_datasets() -> list[Dataset]` | All persisted Datasets in this project | Reads from `dataset_index.json`. (Note: data domain's `list_datasets()` Tier 2 verb wraps this + joins with the experiment-overview active tag.) |
| `log_json(name: str, payload: dict) -> None` | — | Loose-tier project-scoped artifact (for `data_card`, `data_learning` legacy keys still emitted by some code paths) |

`ProjectOverview` (domain type, lives in `project/overview.py`):

```python
@dataclass(frozen=True)
class ProjectOverview:
    schema_version: int = 1
    project_name: str = ""
    created_at: str = ""               # ISO8601
    current_experiment_id: str | None = None
    dataset_count: int = 0
    # ... future fields: Optional with defaults
```

### 6.2 `mlflow.experiment.*`

Experiment-level reads + writes + queries. `experiment_id` defaults to the bound one; pass explicit for cross-experiment ops.

#### 6.2.1 Lifecycle (`experiment/lifecycle.py`)

| Function | Returns |
|---|---|
| `ensure(experiment_id: str \| None = None) -> None` | — |
| `ensure_overview(experiment_id: str \| None = None) -> ExperimentOverview` | `ExperimentOverview` |
| `read_overview(experiment_id: str \| None = None) -> ExperimentOverview \| None` | `ExperimentOverview` |
| `write_overview(overview: ExperimentOverview, experiment_id: str \| None = None) -> None` | — |

**`ensure` / `ensure_overview` idempotent-bootstrap contract** (sub-spec 09 verify-in-02 item,
resolved 2026-05-26): `ensure` creates the MLflow experiment + overview run if **absent**, and
returns the current state if **active**. It sets the `created_by` tag on first creation.

**No auto-restore (the legacy `store.py::_activate_experiment` restore is dropped).** A
**soft-deleted** experiment is archived — `ensure` does **not** silently restore it and does
**not** silently hard-purge it. If a re-`ensure` collides with a soft-deleted same-name
experiment (MLflow reserves the name until restore/hard-delete), MLflow's create fails and that
surfaces as a `StorageError` directing the user to either hard-delete the old one
(`automl experiment delete <id> --apply --hard-delete`, sub-spec 03) or choose a different
`experiment_id`. Resurrecting an archived experiment — inheriting its runs, tags, and lineage —
is exactly the relinkage complexity this design avoids. This is the original open-questions
"intentional cut," kept at full scope. (Soft-delete stays 03's default; `--hard-delete` is the
opt-in for freeing the name.)

`ExperimentOverview` (domain type, lives in `experiment/store.py`):

```python
@dataclass(frozen=True)
class ExperimentOverview:
    schema_version: int = 1
    experiment_id: str = ""
    project_name: str = ""
    created_at: str = ""
    dry_run: bool = False
    # future fields: Optional with defaults

    @classmethod
    def from_dict(cls, payload: dict) -> "ExperimentOverview": ...   # see §8
```

The active dataset for this experiment is NOT a field on `ExperimentOverview` — it has one canonical accessor: `mlflow.experiment.get_active_dataset()` (see §6.2.3). Storing it on the overview dataclass would duplicate the source of truth.

#### 6.2.2 Queries (`experiment/queries.py`)

| Function | Returns |
|---|---|
| `list_trials(experiment_id=None, *, limit=None, status=None) -> list[TrialSummary]` | newest-first |
| `top_n_by_metric(metric: str, n: int = 10, *, ascending: bool = False, experiment_id=None) -> list[TrialSummary]` | sorted |
| `search_trials(filter_string: str, *, experiment_id=None, max_results=1000) -> list[TrialSummary]` | mid-level escape hatch. Scoped to one experiment. Implementation paginates internally. |
| `next_trial_number(*, experiment_id=None) -> int` | next sequential trial number for the experiment (absorbs legacy `_next_trial_number_from_mlflow` / `_run_trial_number`). Sub-spec 08 Q2 carry-back — moving the query to the seam breaks the backward `runner → experiment` import; the runner still *assigns* the number at exec time. |

**`top_n_by_metric` — the `metric` argument is the cross-trial-stable namespaced key, not a bare metric name (sub-spec 11 #3).** The whole locked metric set is logged under `<label>.<metric>` keys (e.g. `holdout.auc`, the namespaced log at `eval/evaluate.py:588`), which is stable across trials regardless of any single trial's *current* primary. Sort/lookup addresses the metric by this `<label>.<metric>` key so a trial reported "missing" is **genuinely uncomputed**, not merely "not this trial's primary." (The bare-`<primary>` metric — legacy `evaluate.py:596` — is a per-trial display convenience read by `TrialSummary.primary_metric_value`, **not** the cross-trial sort key.)

`TrialSummary` (domain type, lives in `trial/types.py`):

```python
@dataclass(frozen=True)
class TrialSummary:
    schema_version: int = 1
    run_id: str = ""
    slug: str = ""
    strategy: str = ""
    status: TrialStatus = TrialStatus.UNKNOWN
    primary_metric_name: str = ""
    primary_metric_value: float | None = None
    started_at: str | None = None         # ISO8601
    ended_at: str | None = None
    parent_run_id: str | None = None
    dataset_hash: str | None = None
    # Added by sub-spec 10 Q3 (additive; all tag/metric-derived → cost-free on the row):
    trial_number: int | None = None
    hypothesis: str = ""
    training_origin: str = ""             # "automl" | "human"
    training_time_s: float | None = None
    n_features: int | None = None

class TrialStatus(str, Enum):
    UNKNOWN = "UNKNOWN"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    KILLED = "KILLED"
```

#### 6.2.3 Datasets (`experiment/datasets.py`)

(Renamed from `snapshots.py` per sub-spec 05 Q1 vocab retirement. Functions + types renamed in lock-step.)

| Function | Returns |
|---|---|
| `list_datasets_for_experiment(experiment_id=None) -> list[DatasetRef]` | All Datasets registered as having been used by trials under this experiment |
| `get_active_dataset(experiment_id=None) -> str \| None` | Active `dataset_id` or None |
| `set_active_dataset(dataset_id: str, experiment_id=None) -> None` | **Pointer-flip only** — sets a tag on the experiment-overview run to mark this Dataset as active for this experiment. Does NOT write the Dataset's data, manifest, or index entry — those are written by the data domain via `mlflow.project.artifacts.write_dataset(...)` when the Dataset is materialized (sub-spec 05 Q3 `materialize()`). |
| `resolve_dataset_id(identity_hash: str, experiment_id=None) -> str \| None` | Identity-hash → `dataset_id` (looks up in the experiment's used-dataset list; returns None if no trial under this experiment has used a Dataset with that identity_hash). For project-scoped resolution (across all experiments) use `mlflow.project.artifacts.resolve_dataset_id` instead. |

**Separation of concerns:** the data domain (`automl.data`) is responsible for *materializing* a Dataset — writing parquet, building the manifest, registering it in the project's `dataset_index.json`. The mlflow seam's `set_active_dataset(dataset_id)` is the *pointer flip* that marks one of those materialized Datasets as the one this experiment uses. This is a deliberate split: writing a Dataset and choosing which Dataset is active are different operations.

`DatasetRef` is defined in `data/contract.py` per sub-spec 05 Q9 (Dataset references are part of the trial data contract surface; the data domain owns the type).

#### 6.2.4 Artifacts (`experiment/artifacts.py`) — loose tier only

Experiment-level artifacts (agent session reports, leaderboard snapshot caches, ad-hoc experiment-scoped JSON) all go through the **loose tier** — same `log_json` / `load_json` / `list_json` pattern used at the trial level. No typed writers at this level; the artifacts here are all in-flux or cache-shaped.

| Function | Notes |
|---|---|
| `log_json(name: str, payload: dict, *, experiment_id=None) -> None` | Loose-tier artifact write to the experiment-overview run |
| `load_json(name: str, *, experiment_id=None) -> dict \| None` | None if not present |
| `list_json(*, experiment_id=None) -> list[str]` | Names of all JSON artifacts on the overview run |

Typed experiment-level artifacts can be promoted later if a shape stabilizes; today there's no demand.

#### 6.2.5 Analytical queries — NO seam placeholder (sub-spec 09 §Q4)

There is **no** `experiment/diagnostics.py` placeholder file (zero placeholder files —
`feedback_extension_points_follow_demand`). The analytical queries split by demand:

- **In scope, view-side** (compose the seam, no new seam primitive): `recent_failures`,
  `strategies_attempted` live in `experiment/views/queries.py`; `compare` in
  `experiment/views/compare.py`. They are realized over `list_trials` / `top_n_by_metric`.
- **Deferred, no file**: `runs_using_strategy`, `runs_in_metric_band` have no caller.
  Recorded as deferred in the living docs; when a real caller appears, add a seam search
  (`search_trials`) + a thin view helper at that time.

### 6.3 `mlflow.trial.*`

Trial-level operations. `run_id` is always explicit — there is no "active trial" implicit state outside the `active()` context manager.

#### 6.3.1 Lifecycle (`trial/lifecycle.py`)

```python
@contextmanager
def active(
    *,
    slug: str,
    strategy: str,
    parent_run_id: str | None = None,
    experiment_id: str | None = None,
) -> Iterator[str]:
    """Open a trial MLflow run and yield its run_id.

    The context manager owns the lifecycle:
      - clean exit  → end with status='FINISHED'
      - exception   → end with status='FAILED' and re-raise

    For lifecycle control outside a context manager (rare; advanced cases like
    long-running async trials or post-failure repair), use start() + end()
    directly — they're siblings to active(), not callable from inside it.
    """

def start(
    *,
    slug: str,
    strategy: str,
    parent_run_id: str | None = None,
    experiment_id: str | None = None,
) -> str:
    """Open a trial run without auto-end. Returns run_id. Caller must call end().

    Use only when the active() context manager doesn't fit (most code should
    use active()).
    """

def end(run_id: str, status: TrialStatus) -> None:
    """Explicitly end a trial run with a specific status. Pair with start()."""
```

#### 6.3.2 Logging (`trial/logging.py`)

Same functions used inside `active()` (with `run_id` from yield) or post-hoc (with `run_id` known ahead of time). No active/post-hoc distinction.

| Function | Notes |
|---|---|
| `log_metric(run_id: str, key: str, value: float, step: int \| None = None) -> None` | |
| `log_metrics(run_id: str, metrics: Mapping[str, float], step: int \| None = None) -> None` | batch |
| `log_param(run_id: str, key: str, value: str) -> None` | |
| `log_params(run_id: str, params: Mapping[str, str]) -> None` | batch |
| `set_tag(run_id: str, key: str, value: str) -> None` | |
| `set_tags(run_id: str, tags: Mapping[str, str]) -> None` | batch |
| `log_json(run_id: str, name: str, payload: dict) -> None` | **loose tier** — no schema, just JSON-serializable |

#### 6.3.3 Reads (`trial/reads.py`)

| Function | Returns |
|---|---|
| `get_details(run_id: str) -> TrialDetails` | full state — params, metrics, tags, artifact paths |
| `get_metrics(run_id: str) -> dict[str, float]` | latest metric values |
| `list_artifacts(run_id: str) -> list[ArtifactRef]` | artifact paths + sizes (no contents) |
| `get_parent_experiment(run_id: str) -> ParentExperimentRef` | the routed experiment owning this run, parsed into typed fields (mode / project / experiment_id). Used by cleanup's trial-delete path to verify the run matches the session's mode and project before acting. Added 2026-05-22 per sub-spec 03 §9.2. |

`TrialDetails` and `ParentExperimentRef` live in `trial/types.py`.

`TrialDetails` (domain type, lives in `trial/types.py`; **fields settled by sub-spec 10 Q1/Q4**):

```python
@dataclass(frozen=True)
class TrialDetails:
    schema_version: int = 1
    run_id: str = ""
    status: TrialStatus = TrialStatus.UNKNOWN
    params: dict = field(default_factory=dict)     # raw
    metrics: dict = field(default_factory=dict)    # raw
    tags: dict = field(default_factory=dict)       # raw
    artifacts: list[ArtifactRef] = field(default_factory=list)
    evaluations: list[EvalResult] | None = None    # None = not loaded (get_details); [] = loaded-empty (show_trial)
    @classmethod
    def from_dict(cls, payload: dict) -> "TrialDetails": ...
```

`get_details(run_id)` returns `TrialDetails` with `evaluations=None` (cheap — no eval-artifact
downloads). The domain verb `trial.show_trial(run_id)` (sub-spec 10) = `get_details` + loading
the eval artifacts into a populated `evaluations` (`EvalResult` is eval's type, sub-spec 07).
`TrialDetails` is independent of `TrialSummary` (no composition); the run→type mapping for both
lives in private seam builder helpers (shared derivation, not shared type — sub-spec 10 Q2).

`ParentExperimentRef` (domain type, lives in `trial/types.py`):

```python
@dataclass(frozen=True)
class ParentExperimentRef:
    """The routed experiment that owns a given MLflow run, parsed into typed fields.

    Returned by `mlflow.trial.get_parent_experiment(run_id)`. Used by cleanup's
    trial-delete path (sub-spec 03 §9.2) to verify the run matches the session's
    mode and project before acting.
    """
    schema_version: int = 1
    mlflow_experiment_id: str = ""    # MLflow's internal id (string like "42")
    mlflow_experiment_name: str = ""  # the routed name, e.g. "payment_routing/baseline-sweep"
    dry_run: bool = False             # parsed from the `dry_run/` prefix in the name
    project_name: str = ""            # parsed from the name
    experiment_id: str = ""           # AutoML's experiment id, parsed from the name

    @classmethod
    def from_dict(cls, payload: dict) -> "ParentExperimentRef": ...
```

#### 6.3.4 Artifacts (`trial/artifacts/<thing>.py`, re-exported flat)

The `__init__.py` re-exports every writer/loader at the `mlflow.trial.artifacts.*` level (no `mlflow.trial.artifacts.eval.write` — see §3.3 import path rule).

**Singleton artifacts (one per trial):**

Note: `proposal/proposal.json` is written by `trial.create()` at trial-create time (sub-spec 10 — Trial domain), NOT by an `mlflow/` writer. The mlflow seam reads it back via the generic artifact-load path; there is no `write_proposal` in this surface.

| Function | Payload type | Lives in |
|---|---|---|
| `write_trial_data_contract(run_id, payload: TrialDataContract) -> TrialDataContractRef` | `TrialDataContract` | `data/contract.py` (renamed from `RunDataContract` per sub-spec 05 Q9) |
| `load_trial_data_contract(run_id) -> TrialDataContract` | | |
| `write_model(run_id, payload: ModelArtifact) -> ModelRef` | `ModelArtifact` | `model/base.py` |
| `load_model(run_id) -> ModelArtifact` | | |
| `write_model_report(run_id, payload: ModelReport) -> ModelReportRef` | `ModelReport` | `model/base.py` |
| `load_model_report(run_id) -> ModelReport` | | |
| `write_feature_importance(run_id, payload: FeatureImportance) -> FeatureImportanceRef` | `FeatureImportance` | `data/features.py` |
| `write_feature_registry(run_id, payload: FeatureRegistry) -> FeatureRegistryRef` | `FeatureRegistry` | `data/features.py` |
| `write_timing(run_id, payload: TimingReport) -> TimingRef` | `TimingReport` | `trial/metadata.py` |
| `write_validation_report(run_id, payload: ValidationReport) -> ValidationReportRef` | `ValidationReport` | `validate/base.py` |
| `write_failure(run_id, message: str, *, kind: str = "") -> None` | — | (no payload type; free text) |
| `write_manifest(run_id, payload: TrialManifest) -> ManifestRef` | `TrialManifest` | `trial/metadata.py` |
| `stage_code_bundle(run_id, code_dir: Path) -> CodeBundleRef` | — | |
| `fetch_code_bundle(run_id) -> Path` | — | unpacks to a temp dir |

**Multi-instance artifacts (N per trial, keyed by label):**

| Function | Payload type | Lives in |
|---|---|---|
| `write_eval(run_id, label: str, payload: EvalResult) -> EvalResultRef` | `EvalResult` | `eval/results.py` |
| `load_eval(run_id, label: str) -> EvalResult` | | |
| `list_eval(run_id) -> list[tuple[str, str]]` | (label, eval_dataset_id) pairs — sourced from MLflow tags, no per-artifact fetch | |
| `write_predictions(run_id, label: str, payload: Predictions) -> PredictionsRef` | `Predictions` | `eval/predictions.py` |
| `load_predictions(run_id, label: str) -> Predictions` | | |
| `list_predictions(run_id) -> list[str]` | label list; symmetric with `list_eval` but no independent eval-dataset identity beyond the eval pairing | |

### 6.4 URL helpers (`mlflow.client.*`)

Public string-builders for the MLflow UI, so every view derives links consistently (the
helpers already exist as `store.py::run_url` / `artifact_url`; this just gives them a seam
home — carry-back from sub-spec 09 §8.1, also relied on by 07/08).

| Function | Returns |
|---|---|
| `run_url(run_id: str) -> str` | `{base}/#/experiments/{mlflow_experiment_id}/runs/{run_id}` — resolves the tracking base + numeric experiment id from bound state |
| `artifact_url(run_id: str, artifact_path: str) -> str` | the run-URL plus `/artifacts/{path}` |

Pure formatting over bound connection state; no MLflow write. The numeric experiment id is
resolved internally (it is not exposed as a public type — see sub-spec 09 §Q5, which drops
the public `experiment_id()` helper).

---

## 7. Artifact writer contract — GCS-then-MLflow

Every typed writer follows the same shape (per §3.5):

```python
def write_eval(run_id: str, label: str, payload: EvalResult) -> EvalResultRef:
    """Persist eval results for this label.

    Order of operations:
      1. Write payload bytes to GCS (atomic via _atomic_gcs_write).
      2. Log the GCS URI + label as MLflow tags on the trial run.

    Failure modes:
      - GCS write fails  → raise StorageError; MLflow untouched.
      - MLflow log fails → raise StorageError; orphan blob in GCS,
                            recoverable via `automl cleanup --scope trial`.
    """
    _validate_label(label)
    b = bound()
    gcs_root = _routing.bucket_uri_for(kind="run_bulk", run_id=run_id) + f"eval/{label}/"

    # Step 1: GCS (atomic — partial rollback handled by helper)
    gcs_uri = _atomic.write(
        bucket=b.bucket,
        primary=(gcs_root + "results.json", payload.to_json_bytes()),
        # multi-step writes pass additional secondaries; for results.json there are none
    )

    # Step 2: MLflow commit
    set_tag(run_id, tags.eval_results_uri_tag(label), gcs_uri)
    set_tag(run_id, tags.eval_dataset_id_tag(label), payload.eval_dataset_id)

    return EvalResultRef(uri=gcs_uri, run_id=run_id, label=label)
```

**`_atomic.write(bucket, primary, secondary=...)`** is the shared partial-write helper. Writes `primary` first; for each `secondary`, writes after; if any secondary fails, deletes everything already written and re-raises as `StorageError`. Used by every multi-step GCS write (predictions writes parquet + manifest in two steps; this helper makes that atomic).

---

## 8. Schema strategy — additive-only + version placeholder + `from_dict` loader

Every typed artifact has a `schema_version: int = 1` field from day one. The rules (§3.6):

1. **Every typed schema starts with `schema_version: int = 1`.** Never removed.
2. **New fields are added with `Optional[...] = None` or default values.** Never renamed; never removed; never type-changed.
3. **Loaders use `from_dict(payload)` — they filter unknown keys.** Bare `Cls(**payload)` raises `TypeError` if the payload has fields the current class doesn't know about (i.e. a newer-version artifact being read by older code). The `from_dict` classmethod strips unknown keys before constructing, so older code can still read newer payloads — losing only the unknown fields, not crashing.
4. **Breaking changes (if ever needed) bump `schema_version` and add a version-dispatching loader at that moment.** No speculation today.

The `from_dict` pattern (one helper applied by every typed schema):

```python
import dataclasses
from typing import ClassVar

@dataclass(frozen=True)
class EvalResult:
    schema_version: int = 1
    label: str = ""
    eval_dataset_id: str = ""
    primary_metric_name: str = ""
    primary_metric_value: float = 0.0
    secondary_metrics: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict) -> "EvalResult":
        """Forward-compatible loader. Strips unknown keys before construction."""
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in known})
```

Every typed payload class follows this exact `from_dict` shape. Loaders in `mlflow/<noun>/artifacts/*.py` call `EvalResult.from_dict(payload)` — never bare `EvalResult(**payload)`.

Example schema growth without breaking (both directions):

```python
# v1 (today)
@dataclass(frozen=True)
class EvalResult:
    schema_version: int = 1
    label: str = ""
    eval_dataset_id: str = ""
    primary_metric_name: str = ""
    primary_metric_value: float = 0.0
    secondary_metrics: dict[str, float] = field(default_factory=dict)
    @classmethod
    def from_dict(cls, payload): ...   # filters unknowns

# v1.5 (later — additive)
@dataclass(frozen=True)
class EvalResult:
    # ... same fields ...
    confidence_intervals: dict[str, tuple[float, float]] | None = None   # NEW
    explanation: str | None = None                                       # NEW
    @classmethod
    def from_dict(cls, payload): ...
```

Forward-read (old → new): v1.5 reader loading a v1 artifact → missing fields default to None. ✓
Backward-read (new → old): v1 reader loading a v1.5 artifact → `from_dict` strips `confidence_intervals` and `explanation` before construction. ✓ (Lossy but doesn't crash.)
No schema_version bump needed for additive changes.

For the loose tier (anything written via `trial.log_json`), there is no schema and no version field. The contract is just "must be JSON-serializable." Promotion from loose to typed is a separate refactor when the shape stabilizes.

---

## 9. Level contract — what lives at each level

This table is the discovery answer. A DS reading this knows what's queryable at each level.

### Project level

**Storage:** one MLflow experiment named `<project_name>/overview` containing one MLflow run named `overview`.

| Kind | Key | Purpose |
|---|---|---|
| Tag | `automl.project_name` | Project's folder name |
| Tag | `automl.created_at` | ISO timestamp of overview creation |
| Tag | `automl.current_experiment_id` | Most-recently-active experiment (advisory) |
| Artifact | `dataset_index.json` | All Datasets materialized for this project (identity_hash → dataset_id + paths) |
| Artifact | `<dataset_id>/profile/...` | EDA profile per Dataset (data_card.json, charts, observations) — per sub-spec 05 Q5 |

**Out of scope for this sub-spec:** project-level "learning" artifacts (golden features, weak features, accumulated cross-experiment observations). Today's `store.py` has `write_learning_cache` / `learning_feature_payloads` for these — they remain in `automl_legacy/` and do not move into the new `mlflow/` surface. They are not well-thought-out yet; if/when they become first-class, a future sub-spec defines them.

**Queryable:**
- `mlflow.project.read_overview()` — full overview as `ProjectOverview`
- `mlflow.project.list_experiments()` — all experiment_ids (see §13 for the implementation note on how this is populated)

### Experiment level

**Storage:** one MLflow experiment named `<project_name>/<experiment_id>` containing one `overview` run + N trial runs.

**Overview run (one per experiment):**

| Kind | Key | Purpose |
|---|---|---|
| Tag | `automl.experiment_id` | This experiment's id |
| Tag | `automl.active_dataset_hash` | Pinned Dataset for trials in this experiment |
| Tag | `automl.active_dataset_name` | Friendly name of the active Dataset |
| Tag | `automl.dry_run` | `"true"`/`"false"` |
| Tag | `automl.created_at` | ISO timestamp |
| Artifact | `agent/sessions/<session_id>/report.json` | Reconciled session reports (loose tier) |
| Artifact | `leaderboard_snapshot.json` | Cached leaderboard view (typed) |

**Trial runs (many — see Trial level).**

**Queryable:**
- `mlflow.experiment.read_overview()` — `ExperimentOverview`
- `mlflow.experiment.list_trials()` — all trials
- `mlflow.experiment.top_n_by_metric("auc", n=10)` — sorted
- `mlflow.experiment.search_runs(filter_string="…")` — raw MLflow filter
- `mlflow.experiment.list_datasets_for_experiment()` — Datasets used by trials under this experiment (renamed from `list_snapshots` per sub-spec 05 Q1)

### Trial level

**Storage:** one MLflow run inside the experiment's MLflow experiment.

**Tags:**

| Key | Purpose |
|---|---|
| `automl.trial.slug` | DS-meaningful name |
| `automl.trial.strategy` | "baseline", "tree", "ensemble", etc. |
| `automl.trial.parent_run_id` | If forked from another trial |
| `automl.trial.dataset_hash` | Which Dataset was used for training |
| `automl.trial.status` | UNKNOWN / RUNNING / FINISHED / FAILED / KILLED. **Clean cut from legacy:** old runs used lowercase strings (`"success"`, `"failed"`); new runs use the uppercase `TrialStatus` enum values. Queries against legacy runs will not match new status filters — intentional, acceptable per the no-back-compat-for-persisted-state rule. |
| `automl.trial.eval.<label>.eval_dataset_id` | Eval dataset identity per label. Powers `list_eval(run_id)` without artifact fetching. |
| `automl.trial.eval.<label>.uri` | GCS URI of the eval-results artifact for this label |

**Params:** training hyperparameters (whatever the model logged via `log_param`)

**Metrics:**
- Model metrics: `auc`, `log_loss`, etc. — sourced from `EvalResult`
- Agent metrics (post-hoc): `agent.proposer_seconds`, `agent.coder_seconds`, `agent.runner_execution_seconds`, `agent.tool_calls`

**Artifacts:**

| Path | Tier | Notes |
|---|---|---|
| `proposal/proposal.json` | typed | Proposal contract (`agent/proposal.py` — sub-spec 11) |
| `model.pkl` | typed | Cloudpickled `BaseModel` instance |
| `model_report.json` | typed | Model self-report (training summary) |
| `data_contract.json` | typed | `TrialDataContract` — the training data the trial saw (renamed from `DataContract` per sub-spec 05 Q9) |
| `feature_importance.json` | typed | `FeatureImportance` |
| `feature_registry.json` | typed | `FeatureRegistry` |
| `timing.json` | typed | `TimingReport` |
| `validation_report.json` | typed | `ValidationReport` |
| `failure.log` | loose | Free-form (only if failed) |
| `manifest.json` | typed | `TrialManifest` — index of what's in this trial |
| `code_bundle.tar` | binary | Code snapshot at trial start |
| `eval/<label>/results.json` | typed (multi) | `EvalResult` per label |
| `eval/<label>/predictions.parquet` | typed (multi) | `Predictions` per label |
| `debug/<anything>.json` | loose | Runner-side `trial.log_json` writes |
| `agent/proposer/report.json` | loose | Post-hoc by hook |
| `agent/coder/report.json` | loose | Post-hoc by hook |
| `agent/coder/tool_events.json` | loose | Post-hoc by hook |

**Queryable:**
- `mlflow.trial.get_details(run_id)` — `TrialDetails`
- `mlflow.trial.get_metrics(run_id)` — `dict[str, float]`
- `mlflow.trial.list_artifacts(run_id)` — `list[ArtifactRef]`
- `mlflow.trial.artifacts.list_eval(run_id)` — labels
- `mlflow.trial.artifacts.load_eval(run_id, label)` — `EvalResult`
- `mlflow.trial.artifacts.load_predictions(run_id, label)` — `Predictions`
- ... (every `load_*` for typed artifacts)

---

## 10. Multi-instance artifacts (eval + predictions)

Why eval and predictions are multi-instance: a single trained model is evaluated against multiple eval-snapshots. Conventional defaults are `label="train"` and `label="test"`. Additional labels: anything the DS or system attaches (e.g., `"q2_2026_holdout"`, `"week_42_oot"`, or a snapshot-derived name).

**Path layout per trial:**

```
eval/
├── train/
│   ├── results.json          ← EvalResult schema
│   └── predictions.parquet   ← Predictions schema
├── test/
│   ├── results.json
│   └── predictions.parquet
└── <any_label>/
    ├── results.json
    └── predictions.parquet
```

**`EvalResult.eval_dataset_id`** carries the canonical content-hash identity. The label is the friendly key for retrieval; the eval-dataset identity is for downstream queries like "which trials were evaluated against eval dataset X" (deferred analytics — no placeholder file, sub-spec 09 §Q4).

**Adding a new eval label** is a new `write_eval(run_id, label=..., payload=...)` call — creates a new folder under `eval/`. No index update.

**Listing what evals exist** is `list_eval(run_id)` — returns `list[tuple[str, str]]` of `(label, eval_dataset_id)` pairs. The `eval_dataset_id` is fetched from MLflow tags (set at write time by `write_eval`), so listing does NOT require loading each artifact — one MLflow call, zero GCS reads. Callers can filter by `eval_dataset_id` without fetching anything. (Implementation note in §13: this is backed by tag scans, not by walking GCS or by loading each results.json.)

---

## 11. Error model

**One exception type:** `StorageError` (in `errors.py`, subclass of `AutoMLError`). Wraps the underlying backend exception via `__cause__`.

```python
class StorageError(AutoMLError):
    """A persistence backend (MLflow tracking, GCS) refused or failed.

    Wraps the underlying backend exception via __cause__ so tracebacks stay intact.
    Domain code catches this type; it never imports mlflow.exceptions or
    google.cloud exceptions directly.
    """
```

**What raises `StorageError`:**
- MLflow API errors (tracking server unreachable, auth failed, run not found in a write context)
- GCS errors (bucket access denied, network failure, partial write that couldn't be rolled back)

**What raises Python-native exceptions:**
- `ValueError` / `TypeError` — programmer errors (bad args, malformed names, regex violations)

**What returns empty / `None`:**
- "No data" cases — `list_trials` on an empty experiment returns `[]`; `read_overview` when not yet written returns `None`; `get_active_dataset` when none pinned returns `None`. Not errors.

Subtypes (`StorageConnectionError`, `StorageNotFoundError`, `StorageWriteError`) are deferred per §3.6 — added only when domain code needs to branch on the error kind, which it doesn't today.

---

## 12. Carry-back to sub-spec 01

**The contract:** **anywhere the session changes, `mlflow.bind()` is re-fired in lock-step.** All three session-affecting entry points in sub-spec 01 must do this — settled, not deferred:

- `automl.use_project(...)` — call `mlflow.bind(...)` after setting the session contextvar.
- `automl.update_session(**kwargs)` — call `mlflow.bind(...)` again with the new field values after `dataclasses.replace`.
- `automl.active_session(name, ...)` (context manager) — call `mlflow.bind(...)` on entry; the existing Session contextvar Token-and-restore on exit transitively scopes the bind, so no separate mlflow Token bookkeeping is needed (the Session restoration triggers a re-bind to the prior session's values).

Sketch for `update_session`:

```python
def update_session(**kwargs: Any) -> Session:
    current = session()
    new = dataclasses.replace(current, **kwargs)
    _ACTIVE_SESSION.set(new)
    _bind_mlflow_for(new)   # shared helper, enumerates every Session field that maps to bind()
    return new

def _bind_mlflow_for(s: Session) -> None:
    """Single source of truth for translating a Session into mlflow.bind() args.
    Enumerates every bound field — if you add one, you only change this helper."""
    from automl import mlflow as _mlflow
    _mlflow.bind(
        tracking_uri=s.config.mlflow_tracking_uri,
        bucket=s.config.gcs_bucket,
        gcs_prefix=s.config.gcs_prefix,
        project_name=s.config.project_name,
        experiment_id=(
            s.experiment_id
            if s.experiment_id is not None
            else (s.config.run_config.experiment_id if s.config.run_config is not None else None)
        ),
        dry_run=s.dry_run,
        namespace=s.namespace,    # Session.namespace, fed by the top-level --namespace flag (sub-spec 01); "" = real
    )
```

Centralizing the Session → `bind()` translation in one helper avoids the bug class where adding a new bound field requires three coordinated edits.

---

## 13. What this sub-spec defers

To implementation-time decisions (not design):

1. **Manifest-driven artifact listing** — instead of walking the MLflow artifact API for `list_artifacts(run_id)` (which has network cost), use the trial's `manifest.json` artifact as an index. Write the manifest on trial end; readers consult it first; fall back to MLflow scan if manifest is missing or stale. Implementation detail of `trial/reads.py`.

2. **Tag-key namespacing convention** — the exact format for compound tag keys (e.g., `automl.trial.eval.<label>.eval_dataset_id` — dot-separated nesting). Pick a single convention and apply consistently across `tags.py` constants.

3. **Internal pagination for `list_trials` / `search_trials`** — MLflow's `search_runs` paginates at 1000 results. Implementations of `list_trials` and `search_trials` must internally follow `page_token` and return complete results for the requested scope. **Not exposed in the public surface** — the caller never sees pagination. (A future optimization may add explicit pagination if response sizes become unwieldy.)

4. **`mlflow.project.list_experiments()` data source** — today's `find_prior_experiment` enumerates raw MLflow experiments and filters by name prefix `<project>/`. The new `list_experiments()` does the same internally (no separate index needed). When/if a project-level experiment index becomes useful (e.g. for `list_experiments_with_metadata`), it can be added — but not before real demand.

5. **`namespace` (was `route_namespace`) source of value — RESOLVED (final pass 2026-05-27).** It is fed by a **top-level `--namespace` flag** on `automl` (+ an env var for subprocess inheritance), mapping to `Session.namespace` (sub-spec 01), defaulting to `""` (real). Renamed `route_namespace` → `namespace` (clean cut). It is a **full-universe isolation dimension** — segregates MLflow experiment names + GCS prefixes + local trial sandbox dirs (sub-spec 08 path) — orthogonal to dry_run, for full-fidelity QA/test sandboxes the user can clean up cleanly. (No longer deferred.)

6. **Predictions overwrite / repair paths** — today's `predictions.py` has a staging-and-promote pattern for `overwrite=True` and a 3-state repair path for re-entry. The new `_atomic.py` partial-rollback covers append; overwrite/repair patterns are added to `_atomic.py` when the migration ports `write_predictions`. The cleanup verb (sub-spec 03) handles the "repair" case by removing orphans first.

7. **Hook subprocess `bind()` bootstrap** — `hooks/agent_timeline.py` (and its successor thin stub) is a subprocess called by Claude Code with no `use_project()` bootstrap. The stub MUST call `mlflow.bind(...)` itself from environment values before any `mlflow.*` call. Implementation detail of `agent.timeline.handle_event`.

8. **Multi-process write coordination** — when two trial subprocesses log to the same experiment simultaneously, MLflow handles run-level isolation but the experiment-overview run has a single tag namespace. Last-write-wins. Deferred if it becomes a problem.

To future sub-specs:

9. **Cleanup orchestration** sub-spec (Priority 3 in structural spec §15.1) — defines the cascading delete order for orphan blobs and the dry-run scope semantics.

10. **Analytical queries** (sub-spec 09 §Q4) — `recent_failures` / `strategies_attempted` / `compare` are **in scope** as view-side composers over the seam (`experiment/views/`); `runs_using_strategy` / `runs_in_metric_band` are deferred with **no placeholder file** (added on real demand).

11. **Proposer-context assembly** — the composite that today's `loop_context/proposer_packet.py` builds via `store.get_context` (a 500-line aggregator combining leaderboard + failures + per-snapshot profile metadata + URLs) moves to **`agent/proposer_context.py`** (domain side). It composes mlflow.* building blocks rather than being a single mlflow function. Concrete shape settled in sub-spec 11 (Agent domain) — the experiment mega-domain was split into `experiment/` + `trial/` + `agent/` during sub-spec 09.

Out of scope for the refactor entirely:

12. **Project-level "learning" subsystem** — golden features, weak features, accumulated cross-experiment observations, learning cache JSONs. Today's `store.py` has `write_learning_cache` / `learning_feature_payloads` for these; they remain in `automl_legacy/` and do NOT migrate. If/when this becomes first-class, a future sub-spec defines it from scratch.

---

## 14. Concrete examples

### Example A — bootstrap + survey

```python
import automl

automl.use_project("payment_routing")           # also calls mlflow.bind()

overview = automl.mlflow.project.read_overview()
print(f"current experiment: {overview.current_experiment_id}")

trials = automl.mlflow.experiment.list_trials()
for t in trials[:5]:
    print(f"  {t.slug:30s}  {t.primary_metric_name}={t.primary_metric_value:.3f}  {t.status}")
```

### Example B — running a trial (runner code)

```python
def run_trial(trial_dir: Path) -> TrialResult:
    s = automl.session()
    proposal = _load_proposal(trial_dir)

    with automl.mlflow.trial.active(slug=proposal.slug, strategy=proposal.strategy) as run_id:
        automl.mlflow.trial.set_tag(run_id, tags.TRIAL_STRATEGY_TAG, proposal.strategy)

        # Train
        df_train, df_test = automl.data.load_active_dataset()
        model = _instantiate_model(trial_dir)
        model.fit(df_train, registry=...)
        automl.mlflow.trial.log_params(run_id, model.hyperparameters)

        # Eval against train + test
        for label, df in (("train", df_train), ("test", df_test)):
            y_pred = model.predict(df)
            preds = Predictions(label=label, eval_dataset_id=..., y_pred=y_pred)
            results = automl.eval.compute(y_pred, df, eval_dataset_id=...)
            automl.mlflow.trial.artifacts.write_predictions(run_id, label=label, payload=preds)
            automl.mlflow.trial.artifacts.write_eval(run_id, label=label, payload=results)
            automl.mlflow.trial.log_metric(run_id, f"{results.primary_metric_name}_{label}",
                                            results.primary_metric_value)

        # Other artifacts
        automl.mlflow.trial.artifacts.write_trial_data_contract(run_id, payload=DataContract(...))
        automl.mlflow.trial.artifacts.write_feature_importance(run_id, payload=...)
        automl.mlflow.trial.artifacts.write_model(run_id, payload=...)

        # Loose-tier debug
        automl.mlflow.trial.log_json(run_id, "debug/training_history.json",
                                      payload={"epoch_losses": [...]})

    return TrialResult(run_id=run_id, status="FINISHED", ...)
```

### Example C — post-hoc hook reconciliation (subprocess, no `use_project()`)

```python
# hooks/agent_timeline.py — thin stub. Runs as a subprocess spawned by Claude Code,
# so there is no automl.use_project() bootstrap. The stub calls bind() directly
# from env / event values.

def handle_subagent_stop(event: dict) -> None:
    import automl
    from automl import mlflow

    # Bootstrap the persistence layer from env (no session needed for post-hoc work)
    mlflow.bind(
        tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
        bucket=os.environ["GCS_BUCKET"],
        gcs_prefix=os.environ["GCS_PREFIX"],
        project_name=event["project_name"],
        experiment_id=event["experiment_id"],
        dry_run=event.get("dry_run", False),
    )

    # Delegate the actual reconciliation logic to the library
    automl.agent.timeline.handle_event(event)


# inside automl.agent.timeline (library code):
def handle_event(event: dict) -> None:
    run_id = event["trial_run_id"]
    automl.mlflow.trial.log_metric(run_id, "agent.coder_seconds", event["duration"])
    automl.mlflow.trial.log_json(run_id, "agent/coder/report.json", event["report"])
```

The hook stub stays thin (the bind + delegate is ~10 lines) and the real logic lives in the library where it's testable.

### Example D — DS exploring a trial in a notebook

```python
import automl
automl.use_project("payment_routing")

# Find the best trial
best = automl.mlflow.experiment.top_n_by_metric("auc", n=1)[0]
print(f"best: {best.slug} ({best.primary_metric_value:.3f})")

# Look at its details
details = automl.mlflow.trial.get_details(best.run_id)
print(f"params: {details.params}")

# Load typed artifacts
eval_train = automl.mlflow.trial.artifacts.load_eval(best.run_id, label="train")
eval_test  = automl.mlflow.trial.artifacts.load_eval(best.run_id, label="test")
print(f"train auc: {eval_train.primary_metric_value:.3f}")
print(f"test auc:  {eval_test.primary_metric_value:.3f}")

# What other eval labels exist?
all_labels = automl.mlflow.trial.artifacts.list_eval(best.run_id)
print(f"evaluated against: {all_labels}")

# Load the predictions DataFrame
preds = automl.mlflow.trial.artifacts.load_predictions(best.run_id, label="test")

# Load the model
model = automl.mlflow.trial.artifacts.load_model(best.run_id)

# Reach for the raw client (escape hatch — rare)
client = automl.mlflow.client.raw()
# do anything not wrapped...
```

---

## Sub-spec status

This sub-spec is complete. Implementation can proceed against this contract. The next sub-spec is **Sub-spec 03 — Cleanup orchestration** (§15.1 Priority 3 in the structural spec).
