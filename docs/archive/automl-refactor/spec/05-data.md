# 05 — Data Domain

**Status:** approved 2026-05-24 (after three-agent review + holistic cross-doc consistency pass; carry-backs applied to 00/01/02)
**Parent:** [`00-structural-design.md`](00-structural-design.md) §8.2
**Related:** [`01-project-context.md`](01-project-context.md), [`02-mlflow-seam.md`](02-mlflow-seam.md), [`03-cleanup.md`](03-cleanup.md), [`04-validate.md`](04-validate.md)

---

## 1. Scope

`automl/data/` owns:
- Dataset identity + materialization (full normalized DataFrame + FeatureRegistry as immutable parquet on GCS).
- Data sources (one file per source type under `sources/`; `DataSource` is the Tier 3 anchor for source-side variation).
- The orchestration pipeline (`DataPipeline`; a second Tier 3 anchor for orchestration-side variation).
- Split definitions (`Splits`) and per-slice loading via pyarrow filter push-down.
- The deterministic data profiler (`data/profile.py`).
- The feature lifecycle catalog (`FeatureRegistry`).
- The trial-level data contract (`TrialDataContract`).

This sub-spec settles the *interface* for these. Implementation details (line-by-line code changes) belong to the implementation plan that comes after all sub-specs are done.

---

## 2. What we have today

Legacy source tree now under `automl_legacy/data/` (originally `automl_dev/automl/data/`, with
the FeatureRegistry living in `core/`):

| Today's file | LOC | Holds |
|---|---|---|
| `data/__init__.py` | 10 | re-exports |
| `data/spec.py` | 56 | `DataSpec` |
| `data/sources.py` | 308 | `DataSource` ABC + `SnowflakeSource` / `LocalCSVSource` / `GCSParquetSource` |
| `data/adapters/{snowflake,local_csv,gcs_parquet}.py` | ~75 total | `SnowflakePipeline` / `LocalCSVPipeline` / `GCSParquetPipeline` — legacy DataPipeline-subclass wrappers (constructor sugar; no real overrides) |
| `data/pipeline.py` | 1262 | `DataPipeline` (12-step waterfall) + `DataPreview`; module-level helpers (`read_sql_file`, `get_df_from_snowflake`, GCS shims) |
| `data/loader.py` | 39 | `build_pipeline(ctx)` |
| `data/snapshot.py` | 518 | `SnapshotIdentity`, `LoadedDataSnapshot`, manifest build/validate, identity composition, GCS paths, split-view helpers |
| `data/snapshots.py` | 387 | `LoadedSnapshot`, `SnapshotSummary`, `SnapshotIndex`, `load_snapshot`, `list_snapshots` |
| `data/run_snapshot.py` | 159 | `load_data_snapshot_for_run` (load by trial run id) |
| `data/prepare.py` | 84 | thin CLI shim around `pipeline.prepare_data()` |
| `data/split.py` | 137 | `HashKey`, `add_split_id`, `hash_key_columns`, `hash_key_report`, `split_report` |
| `data/contract.py` | 322 | `RunDataContract`, `Shape`, `SplitShape`, `RunRef`, `SnapshotRef`, `SplitContract`, `validate_run_data_contract` |
| `core/feature_registry.py` | 649 | `FeatureRegistry`, `FeatureEntry` |
| `profile/core.py` | 479 | pure profiler: stats checks + matplotlib charts + observations |
| `profile/snapshot.py` | 255 | MLflow-publishing wrapper |

Smells the sub-spec resolves:
- "Snapshot" used as both the immutability concept and the user-facing noun — §5 calls this drift; current code has it everywhere.
- Two parallel "loaded" types (`LoadedSnapshot` artifact-native, `LoadedDataSnapshot` with eager train/test) because the pipeline eagerly splits.
- Eager train/test split → leakage risk (model code receives a tuple with both halves in memory) AND opinionated naming (only `train`/`test`).
- `DataPipeline` subclassed by `data/adapters/*.py` for source-selection sugar, conflating two axes (source vs orchestration).
- `null_drop_threshold` / `constant_drop_threshold` / `dry_run_rows` declared in three places (DataSpec field, DataPipeline class attr, DataPipeline ctor kwarg).
- Profile publishing imports MLflow directly from `data/profile/snapshot.py` (violates structural spec §9.1).
- Hardcoded train/test in `SplitShape` and `SplitContract._ranges_to_dict`.
- `RunDataContract.run.run_id` and `trial_id` were conflated in the audit notes, but runner
  spec 08 Q4 corrected this: `trial_id` is the human ordered id (`<number>_<slug>`) and `run_id`
  is the MLflow UUID. Both survive on `TrialRef`.
- `prepare_event_id` carried as durable identity for materialization events with no consumer beyond logging.

---

## 3. Folder shape (final)

```
automl/data/
├── __init__.py            ← Tier 2 re-exports (verbs + types)
├── spec.py                ← DataSpec (config, frozen)
├── pipeline.py            ← DataPipeline (Tier 3: orchestration-side override)
├── features.py            ← FeatureRegistry, FeatureEntry (moved from core/)
├── split.py               ← hashing MECHANISM only: HashKey, hash_key_columns, add_split_id, split_report. (Splits CONFIG type lives in project/run_config.py — see Q8.)
├── contract.py            ← TrialDataContract + TrialRef + DatasetRef + SliceContract + validators
├── dataset.py             ← Dataset, LoadedDataset, LoadedSlice, ComponentHashes, DatasetIndex
├── registry.py            ← list_datasets, load_dataset, load_dataset_by_id, load_dataset_by_trial (the "read" verbs)
├── profile.py             ← deterministic profiler (pure functions + Profile type + profile() / get_profile() verbs)
└── sources/
    ├── __init__.py
    ├── base.py            ← DataSource ABC (Tier 3: source-side override)
    ├── snowflake.py
    ├── local_csv.py
    └── gcs_parquet.py
```

**Gone from today:**
- `data/adapters/` (legacy DataPipeline subclasses — constructor sugar with no real overrides)
- `data/snapshot.py` (split across new files; vocabulary retired)
- `data/snapshots.py` (replaced by `data/registry.py`)
- `data/run_snapshot.py` (folded into `data/registry.py:load_dataset_by_trial`)
- `data/loader.py` (private `_build_pipeline` helper inside `pipeline.py`)
- `data/prepare.py` (CLI wrapper; new CLI calls `materialize()` directly)
- `data/profile/` subfolder collapsed to single `data/profile.py`
- `core/feature_registry.py` (moved to `data/features.py`)

**New / updated MLflow-layer touchpoints** (live in `automl/mlflow/`, not in `data/` — paths match sub-spec 02 §4 folder layout):
- `mlflow/project/artifacts.py` — already houses project-scoped artifacts per sub-spec 02 §4 (lists `dataset_index`, `data_card`, `data_learning`). Profile writers (`write_profile` / `read_profile`) land here since profiles are project-scoped (1:1 with Datasets per §5.1). Dataset manifest + index writers + dataset-id resolution algorithm also live here (was `mlflow/store.py:read_snapshot_index` / `resolve_snapshot_name` today).
- `mlflow/experiment/snapshots.py` → **renamed `mlflow/experiment/datasets.py`** per Q1 vocab retirement. Function signatures rename `list_snapshots` → `list_datasets_for_experiment`, `resolve_snapshot` → `resolve_dataset_id`; existing `get_active_dataset` / `set_active_dataset` stay. Returns `DatasetRef` (not today's `SnapshotRef`).
- `mlflow/trial/artifacts/data.py` — houses `write_trial_data_contract` / `load_trial_data_contract` per sub-spec 02 §6.3.4. Payload type is `TrialDataContract` per Q9.

---

## 4. Decisions

### Q1 — Vocabulary: "snapshot" retires as a noun

`Dataset` is the user-facing noun (already settled by structural spec §5). Sub-spec 05 makes the retirement total at the code level: every `snapshot`-token in `data/` becomes `dataset`. Class names, field names, file names, functions, constants, GCS path components, MLflow tag keys — all renamed.

"Snapshot" survives only as informal docstring language describing what a Dataset *is* (an immutable point-in-time view). It is never an identifier anyone imports or constructs.

| Today | Proposed |
|---|---|
| `SnapshotIdentity` | folded into `Dataset` + `ComponentHashes` |
| `LoadedSnapshot` / `LoadedDataSnapshot` | `LoadedDataset` (full) + `LoadedSlice` (one slice) |
| `SnapshotSummary` | folded into `Dataset` (Q4 — DatasetIndex carries Dataset entries directly) |
| `SnapshotIndex` | `DatasetIndex` |
| `snapshot_id`, `snapshot_name` | `dataset_id` (single name) |
| `snapshot_identity_hash`, `snapshot_hash8` | `identity_hash`, `hash8` (context = Dataset) |
| `snapshot_gcs_paths` | `Dataset.data_gcs_uri` / `registry_gcs_uri` / `manifest_gcs_uri` (properties) |
| `load_snapshot`, `list_snapshots`, `load_data_snapshot_for_run` | `load_dataset`, `list_datasets`, `load_dataset_by_id`, `load_dataset_by_trial` |
| `SNAPSHOT_HASH8_RE`, `SNAPSHOT_NAME_RE` | `DATASET_HASH8_RE`, `DATASET_ID_RE` |
| GCS path `data/snapshots/v<n>_<hash8>/` | `data/datasets/v<n>_<hash8>/` |
| MLflow tag `data.snapshot_id` | `data.dataset_id` |
| MLflow tag `data.snapshot_identity_hash` | `data.identity_hash` |

Clean cut, no back-compat for old MLflow runs / GCS paths / tag values (per `feedback_no_back_compat`).

### Q2 — Public extensibility: two Tier 3 anchors via composition (not inheritance)

Two orthogonal extension axes, each anchored on its own ABC/class:
- `DataSource` (`data/sources/base.py`) — source-side variation: how raw rows arrive, what artifacts a source emits, source-identity payload.
- `DataPipeline` (`data/pipeline.py`) — orchestration-side variation: normalization rules, registry construction, split logic, snapshot policy, anything between raw load and parquet materialization.

Both are Tier 2 exports. Both subclassable. Composition assembles them through `DataSpec`:

```python
DataSpec(source=SnowflakeSource(...), pipeline_cls=MyPipeline, ...)
```

**Deleted:** `data/adapters/{snowflake,local_csv,gcs_parquet}.py` — legacy `*Pipeline` constructor-sugar classes that subclassed DataPipeline only to forward kwargs into a Source. Self-described as "Legacy import wrapper." Per `feedback_no_back_compat`, no `*Pipeline` import alias survives. Users construct sources directly:

```python
# before (status quo)
SnowflakePipeline(base_table=..., base_data_sql=..., training_data_sql=..., raw_target_column=...)

# after
DataSpec(source=SnowflakeSource(base_table=..., base_data_sql=..., training_data_sql=...), ...)
```

This is the composition-over-inheritance correction: source and orchestrator are independent, assembled at config time, not conflated via subclass.

The 13 public override hooks on `DataPipeline` (today's `run_base_sql`, `load_training_data`, `normalize_source_values`, `standardize_columns`, `validate_loaded_data`, `build_feature_registry`, `apply_column_roles`, `infer_dtypes`, `apply_dtypes`, `dedupe`, `apply_quality_filters`, `flag_features`, `split`) carry forward unchanged — they're the documented override surface and the structural spec preserves them.

### Q3 — Public verbs: four free functions, split-at-load (not split-at-store), three load modes

Today's six operations (`build_pipeline`, `prepare_data`, `run`, `preview`, `list_snapshots`, `load_snapshot`) collapse into a focused set of free functions. **The materialized Dataset is `df_data + registry` — no train/test split persisted.** Splits are applied at load time via pyarrow filter push-down so model code only ever receives the slice it asked for (leakage-safe by construction).

| Verb | Returns | Behavior |
|---|---|---|
| `build_dataset()` | `LoadedDataset` | Run the pipeline in memory; no persistence. |
| `materialize(*, refresh_source=False)` | `LoadedDataset` | Run the pipeline; persist parquet + registry + manifest to GCS; register in DatasetIndex. Idempotent (re-use if identity matches). |
| `list_datasets()` | `DatasetIndex` | Read the project's dataset index. |
| `load_dataset(*, split_name=None, split_range=None)` | `LoadedSlice \| LoadedDataset` | Read the active dataset (or its requested slice) for the current session. |
| `load_dataset_by_id(dataset_id, *, split_name=None, split_range=None)` | `LoadedSlice \| LoadedDataset` | Read a specific dataset by id; splits resolved via session config. |
| `load_dataset_by_trial(trial_id, *, split_name=None, split_range=None)` | `LoadedSlice \| LoadedDataset` | Read the dataset a specific trial used; splits resolved via the trial's contract (NOT session config). |

**Naming convention:** the in-memory variant of `materialize` is `build_dataset` (not `preview`). "Preview" misleadingly suggests a partial view (e.g. limit 100); the verb actually runs the full pipeline without persistence. Symmetric: `build_dataset` constructs in memory; `materialize` is `build_dataset` + persist.

**`load_dataset_by_trial` is the audit/reproduction verb.** Trial contract is fully authoritative: `dataset_id` *and* the splits dict come from MLflow, not from config.py. A 6-month-old trial reproduces exactly even after many config edits.

**`load_dataset_by_trial(split_name=...)` resolution rule:** when caller passes `split_name`, it MUST appear as a key in `contract.splits` (the splits dict frozen at trial run time). If not, raise `KeyError` naming the available splits from the contract. config.py / current `session.run_config.splits` is NEVER consulted by this verb. `split_range` is accepted unchanged (explicit ranges don't go through the contract's named-split dict — they're raw range tuples).

**`split_name` ↔ `split_range` are mutually exclusive.** Passing both raises `ValueError`. Passing neither returns a full `LoadedDataset`.

**`split_range` accepts one *or more* disjoint pairs** — its type is `tuple[tuple[int, int], ...]` (the same multi-range shape `Splits` values carry, line below), so `split_range=((80, 90), (95, 100))` is valid, not just one contiguous `(80, 100)`. The reader-side pyarrow filter ORs the buckets. This is required by sub-spec 07 Q1's `split_view` delegation (eval realizes a recipe's buckets via `load_dataset_by_id(of_dataset_id, split_range=buckets)`, and recipes can be non-contiguous). A single pair `((80, 100),)` is the common case; the loader also accepts a bare `(80, 100)` and normalizes it to one pair.

**Range semantics:** half-open `[start, end)` (matches today's `range(lo, hi)` convention at `core/run_config.py:35`). Adjacent ranges sharing an edge (e.g. `(0, 80)` + `(80, 100)`) do NOT overlap. Empty / inverted ranges (`end <= start`) are forbidden by `Splits.__post_init__`. Cross-name overlap is forbidden by the same validator.

**Split definitions are free-form named dicts** (per Q8) held on `RunConfig.splits: Splits`. Defaults preserve today's `{"train": ((0, 80),), "test": ((80, 100),)}` shape. Custom names (e.g. `"fold0_train"`) are supported via `RunConfig.train_split` / `RunConfig.eval_split` name pointers.

### Q4 — Typed objects: `Dataset` / `LoadedDataset` / `LoadedSlice` / `DatasetIndex`

```python
@dataclass(frozen=True)
class ComponentHashes:
    source_identity:  str
    feature_registry: str
    data_content:     str
    schema:           str

@dataclass(frozen=True)
class Dataset:
    # Identity
    id:                str               # "v3_abc12def" (version + hash8)
    identity_hash:     str               # "sha256:..."
    component_hashes:  ComponentHashes
    # Location
    gcs_bucket:        str
    # Provenance
    project_name:      str
    created_at:        str               # ISO timestamp
    # Lineage (identity-contributing)
    source_identity:   dict[str, Any]
    # Schema descriptor
    n_rows:            int
    n_columns:         int
    target_column:     str
    split_id_col:      str
    hash_key:          tuple[str, ...]
    # Persistence
    schema_version:    int = 1

    @property
    def gcs_base_path(self) -> str:    ...   # derived: f"{project_name}/data/datasets/{id}"
    @property
    def data_gcs_uri(self) -> str:     ...
    @property
    def registry_gcs_uri(self) -> str: ...
    @property
    def manifest_gcs_uri(self) -> str: ...

    @classmethod
    def from_dict(cls, payload: dict) -> Dataset: ...
    def to_dict(self) -> dict: ...

@dataclass(frozen=True)
class LoadedDataset:
    dataset:  Dataset
    df:       pd.DataFrame                  # full normalized data, no split
    registry: FeatureRegistry
    @property
    def id(self) -> str:     return self.dataset.id
    @property
    def n_rows(self) -> int: return len(self.df)

@dataclass(frozen=True)
class LoadedSlice:
    dataset:      Dataset
    df:           pd.DataFrame              # ONLY the requested slice rows
    registry:     FeatureRegistry
    split_name:   str | None                # None if loaded by range
    split_ranges: tuple[tuple[int, int], ...]
    @property
    def id(self) -> str:     return self.dataset.id
    @property
    def n_rows(self) -> int: return len(self.df)

@dataclass(frozen=True)
class DatasetIndex:
    datasets:           tuple[Dataset, ...]
    active_dataset_id:  str | None          # set by list_datasets from session.experiment context
    schema_version:     int = 1
    @property
    def active(self) -> Dataset | None: ...
    @classmethod
    def from_dict(cls, payload: dict) -> DatasetIndex: ...
    def to_dict(self) -> dict: ...
    def to_dataframe(self) -> pd.DataFrame: ...
```

**Dropped from today's types** (audited for forward-only use):

| Dropped | Why |
|---|---|
| `prepare_event_id` | Temporal audit covered by `Dataset.created_at` + trial's MLflow `start_time`. No remaining consumer needs the indirection. |
| `run_mode` | Encoded in path / MLflow routing per sub-spec 03's strict isolation. The two universes have separate experiments + GCS prefixes; mode is implicit in where the Dataset lives. |
| `experiment_id` (on Dataset) | Datasets are project-scoped per §5.1 (not experiment-scoped). The experiment context lives on the trial (TrialRef), not on the Dataset. |
| `source_event` | Runtime invocation metadata (e.g. `refresh_source: bool`), not a property of the resulting Dataset. Belongs to the trial's MLflow run, not the Dataset. |
| `gcs_base_path` (as a field) | Derived from `(gcs_bucket, project_name, id)` via routing convention. Stored as a `@property`. |
| `experiment_overview_run_id` (on DatasetIndex) | Derivable from convention (`<project>/overview` per §5.1). mlflow layer resolves on demand. |
| `manifest: dict` raw access | Replaced by typed `Dataset` fields and properties. Every today-consumer of `manifest["gcs"][...]`, `manifest["hashes"][...]`, `manifest["shape"][...]` has a typed accessor. |
| `LoadedDataSnapshot.df_train` / `df_test` | Q3 split-at-load eliminates eager split. Slice loaded via separate `load_dataset(split_name=...)` calls. |

**Composition over flat:** `LoadedDataset.dataset: Dataset` (not field duplication). Convenience pass-throughs for `.id`, `.n_rows`.

**Persistence:** `Dataset` and `DatasetIndex` carry `schema_version: int = 1` + `from_dict` (strip unknown keys) per sub-spec 02 pattern. `LoadedDataset` / `LoadedSlice` / `ComponentHashes` are in-memory only.

**`DatasetIndex.active_dataset_id` population mechanism:** `list_datasets()` reads the persisted index from `mlflow.project.read_dataset_index()` (returns the `datasets` tuple + `schema_version`); then calls `mlflow.experiment.get_active_dataset(experiment_id=session.experiment_id)` (per sub-spec 02 §6.2.3) to fill `active_dataset_id`. If the session has no experiment bound, `active_dataset_id` is `None`. The runtime population is what makes `DatasetIndex.active_dataset_id` "session-context dependent" — the persisted `dataset_index.json` does NOT store an active pointer (active pin is an experiment-overview-run tag, not a project-overview artifact).

### Q5 — Profile: single file in `data/`, MLflow writing lives in mlflow/

`data/profile/` subfolder collapses to a single file `data/profile.py`. Today's split (`core.py` pure + `snapshot.py` MLflow-publishing wrapper) made sense when both lived in `data/`. After moving MLflow writing to `mlflow/artifacts/profile.py` (per structural spec §9.1's "domain code never `import mlflow`"), what's left in the data domain doesn't justify two files. "core" is too abstract; "publish" no longer publishes.

**Three-layer split:**

| Concern | Lives in | Imports `mlflow`? |
|---|---|---|
| Stats + chart functions (pure) | `data/profile.py` (private) | no |
| `profile()` / `get_profile()` verbs (orchestrators) | `data/profile.py` (public) | no — calls `mlflow.artifacts.profile` |
| Writing profile artifacts to MLflow | `mlflow/artifacts/profile.py` | yes (only place) |

**Pluggable checks** — same pattern as sub-spec 04 (validate): named-function list + direct iteration + per-check exception wrapping.

```python
# data/profile.py
_STATS_CHECKS: list[tuple[str, Callable]] = [
    ("basic_stats",        _check_basic_stats),
    ("numeric_summary",    _check_numeric_summary),
    ("categorical_top",    _check_categorical_top),
    ("target_correlation", _check_target_correlation),
]
_CHARTS: list[tuple[str, Callable]] = [
    ("label_distribution",  _chart_label_distribution),
    ("missingness",         _chart_missingness),
    ("correlation_heatmap", _chart_correlation_heatmap),
    ("target_by_segment",   _chart_target_by_segment),
]
```

To add a check: write a function, append to the list. To remove: delete from the list. Per-check exception wrapping so one bad check doesn't tank the whole profile. No `@register` decorator. No plugin system. Project-side custom checks deferred per `feedback_extension_points_follow_demand` (no real demand today).

**`Profile` typed object** (URI-only, no local Path):

```python
@dataclass(frozen=True)
class Profile:
    dataset_id:           str
    target_column:        str
    data_card_uri:        str
    data_observations_uri: str
    profile_manifest_uri: str
    chart_uris:           dict[str, str]    # chart_kind → URI
    created_at:           str
    schema_version:       int = 1
    @classmethod
    def from_dict(cls, payload: dict) -> Profile: ...
    def to_dict(self) -> dict: ...
```

**Public verbs:**
- `profile(dataset_id=None, *, session=None) → Profile` — compute + persist for a Dataset.
- `get_profile(dataset_id=None, *, session=None) → Profile | None` — fetch existing or None.

**No reverse pointer from Dataset to Profile.** The proposer-context aggregator (in `agent/proposer_context.py`, sub-spec 11 territory — the experiment domain was split into experiment/trial/agent) calls both `list_datasets(...).active` and `get_profile(dataset.id)` and merges them. Dataset stays pure (data identity); Profile is its own artifact family; the merge happens at the layer that needs the merged view.

**Bug fix in passing:** today's profile artifacts land under the experiment-overview run; per §5.1 (Datasets are project-scoped, profiles are 1:1 with Datasets) they should land under the project-overview run. New code writes to the right place; per `feedback_no_back_compat`, old paths aren't backwards-readable.

### Q6 — FeatureRegistry: lift, trim learning, add derived/lineage

**Move:** `core/feature_registry.py` → `data/features.py`.

**Trim:** the project-learning subsystem (`golden` / `weak` flags + `apply_learning_flags` / `import_learning_flags` + JSON file parsing) is **out of scope per README** — stays in `automl_legacy/` and does not migrate. ~80 LOC deleted.

**Add:** lineage for transformation-added columns.

```python
@dataclass
class FeatureEntry:
    name: str
    dtype: str
    original_name: str = ""
    null_pct: float = 0.0
    nunique: int = 0
    dominance_pct: float = 0.0
    available: bool = False
    feature: bool = False
    model: bool = False
    target: bool = False
    comments: str = ""

    # NEW
    derived: bool = False                       # True = added by transformation code (not from raw load)
    source_columns: tuple[str, ...] = ()        # for derived cols: which feature cols this came from
```

```python
FLAGS = ("available", "feature", "model", "target", "derived")    # "derived" joins; "golden"/"weak" leave
```

**New helper:**
```python
def add_derived(
    self, name: str, dtype: str,
    source_columns: tuple[str, ...] | list[str],
    *, model: bool = True, comments: str = "",
) -> None:
    """Add a derived column (OHE, log-transform, interaction, etc.).
    Sets derived=True, available=True, model=<param>, populates source_columns.
    Raises ValueError if name already exists.
    Raises KeyError if any source_column is not in the registry."""
```

**Why these and not description / tags / ownership / lifecycle / validation rules / drift / versioning / online-offline / PIT / entities:** researched against modern feature stores (Tecton, Chalk, Feast, Hopsworks, Databricks FS); only `derived` + `source_columns` clear the "real value, real demand today" bar for our context (single-team workspace, single Docker deployment, snapshot-based contract, LLM-agent loop). Descriptions and tags would require manual annotation we don't have; ownership/lifecycle/validation/drift either don't apply (single team, snapshot-immutable) or belong elsewhere (`experiment/views/diagnostics.py` for cross-trial drift).

**Behavior drop — learning-flag injection at materialize time.** Today's `DataPipeline._materialize_dataset` reads `golden_features.json` / `weak_features.json` from MLflow and stamps `golden` / `weak` flags onto the registry *before* writing the parquet (pipeline.py:770-779). With the learning subsystem deferred, this stamping step is removed from the materialize flow. **Effect on materialized content:** the registry CSV no longer carries learning-flag columns → `Dataset.component_hashes.feature_registry` computes differently → the same source data produces a different `Dataset.identity_hash` than today. Per `feedback_no_back_compat` this is acceptable (old runs aren't readable by new code); flagging it explicitly so it's not a surprise downstream.

**Mutability justification.** `FeatureRegistry` stays mutable (only type in this sub-spec that isn't frozen) because: (a) the pipeline waterfall builds it iteratively (build_from_df → infer_dtypes → apply_column_roles → quality_check → etc.) and a copy-and-return pattern at every step would multiply allocations; (b) model code performs `copy.deepcopy(loaded.registry)` and then `reg.add_derived(...)` per derived column — already isolated to a trial-local copy. The frozen `LoadedDataset.registry` reference doesn't make the registry itself immutable; downstream code is expected to deepcopy before mutating. Documented invariant: **the registry returned by `load_dataset*` / `materialize` / `build_dataset` is owned by the caller and may be mutated; the pipeline never re-uses a returned registry.**

**Net diff vs today:**

| Aspect | Today | Proposed |
|---|---|---|
| Location | `automl/core/feature_registry.py` (649L) | `automl/data/features.py` (~580L) |
| `FeatureEntry` fields | 13 (incl. `golden`, `weak`) | 13 (drop 2 learning, add 2 lineage) |
| `FLAGS` tuple | `(available, feature, model, target, golden, weak)` | `(available, feature, model, target, derived)` |
| New method | — | `add_derived(name, dtype, source_columns, ...)` |
| Mutability | mutable | unchanged |
| Persistence | local CSV via `to_dataframe`/`from_dataframe` | unchanged (new columns serialize as JSON arrays for `source_columns`) |

### Q7 — DataSpec: consolidate, fix latent bug, slim DataPipeline ctor

```python
@dataclass(frozen=True)
class DataSpec:
    source:                   DataSource                              # required
    exclude_cols:             tuple[str, ...] = ()
    metadata_cols:            tuple[str, ...] = ()
    pipeline_cls:             type[DataPipeline] = DataPipeline       # Q2: orchestration override
    null_drop_threshold:      float = 0.99                            # drop cols > 99% null
    constant_drop_threshold:  float = 1.0                             # drop strict-constant cols (FIX default)
    dry_run_rows:             int = 10_001                            # row cap when session.dry_run=True
```

Same seven fields as today. Two semantic changes:
- `constant_drop_threshold` default `0.99` → `1.0`. **Latent bug fix:** today's `0.99` disables the check (`_check_constant` at `feature_registry.py:474` returns `{}` whenever threshold `< 1.0`). New default turns the check on, dropping strict-constant columns by default.
- `Sequence` → `tuple` types throughout (already coerced internally; makes the type honest).

**DataPipeline `__init__` slims from 14 kwargs to 3:**

```python
def __init__(
    self,
    spec: DataSpec,
    session: Session,
    *,
    refresh_source: bool = False,
) -> None:
    self.spec = spec
    self.session = session
    self.refresh_source = refresh_source
```

Removed: `raw_target_column` (via `session.task.target`), legacy `base_table`/`base_data_sql`/`training_data_sql` (source-owned; back-compat path deleted), `dry_run` (via `session.dry_run`), `dry_run_rows`/`null_drop_threshold`/`constant_drop_threshold` (via `self.spec.X`), `split` (per Q3 → `RunConfig.splits`), `project_root`/`project_name`/`project_context` (replaced by `self.session` per sub-spec 01).

**Removed DataPipeline class attrs:** `null_drop_threshold`, `constant_drop_threshold`, `dry_run_rows` (all live solely on DataSpec). **Kept:** `split_id_col` (framework constant, not per-project tunable).

**Loader function (`build_pipeline`):** renamed to private `_build_pipeline` (used internally by the verbs from Q3), shrinks from 40L to ~10L:

```python
def _build_pipeline(session: Session, *, refresh_source: bool = False) -> DataPipeline:
    spec = session.config.require_data_spec()        # existing accessor, sub-spec 01 line 85
    return spec.pipeline_cls(spec, session, refresh_source=refresh_source)
```

(Uses sub-spec 01's existing `ProjectConfig.require_data_spec()` accessor — no new `Session.data_spec` API needed. `DataPipeline.__init__` reads `RunConfig` via `session.config.require_run_config()` similarly.)

**User-facing project config code is unchanged.** Cleanup is all internal.

### Q8 — Splits: free-form named dict, reader-agnostic

`project/run_config.py:Split` (hardcoded `train` / `test` slots) → `project/run_config.py:Splits` (free-form named dict). **Location: stays in `project/run_config.py`** (where today's `Split` lives per structural spec §7 line 132 + §8.1), NOT `data/split.py`.

**Why project/, not data/ (corrects an earlier draft):** `RunConfig` lives in `project/` and holds the splits config. The allowed dependency direction is `data → project` (structural spec §8.2 line 282); `project → data` is NOT allowed. If `Splits` lived in `data/` and `RunConfig` (project) held a `splits: Splits` field with a `Splits(...)` default factory, `project` would import `data` at runtime → **dependency cycle**. Keeping `Splits` in `project/run_config.py` (the split *config*) while `data/split.py` keeps the hashing *mechanism* (`HashKey`, `hash_key_columns`, `add_split_id`) matches the structural spec's existing separation (§13.8 line 677) and avoids the cycle. The pyarrow filter helper in `data/registry.py` imports `Splits` from `project` (the allowed direction).

```python
@dataclass(frozen=True)
class Splits:
    """Named ranges over the deterministic 0..99 SPLITID buckets.

    Range convention: half-open [start, end). Adjacent ranges sharing an
    edge (e.g., (0, 80) + (80, 100)) do NOT overlap.

    Pure data structure. Reader-agnostic — knows nothing about parquet,
    pyarrow, CSV, or any specific storage format.
    """
    ranges: dict[str, tuple[tuple[int, int], ...]]

    def __post_init__(self):
        # validates: non-empty ranges dict; non-empty name strings;
        # each name has >=1 range; 0 <= start < end <= 100 (catches empty/inverted);
        # no cross-name overlap (error names both colliding slices)
        ...

    def resolve(self, name: str) -> tuple[tuple[int, int], ...]: ...
    def buckets(self, name: str) -> frozenset[int]:           # reader-agnostic; for in-memory filter via .isin()
        ...

    @classmethod
    def from_dict(cls, payload: dict) -> Splits: ...
    def to_dict(self) -> dict: ...
```

`Splits` validation (the 0..99 bucket grid, half-open range convention) lives in its `__post_init__` — the validation logic doesn't require the class to live in `data/`. `RunConfig` holds a `Splits` instance directly (same module).

**RunConfig change** (sub-spec 01 carry-back):
```python
@dataclass(frozen=True)
class RunConfig:
    experiment_id: str
    splits: Splits = field(default_factory=lambda: Splits({
        "train": ((0, 80),),
        "test":  ((80, 100),),
    }))
    train_split: str = "train"                          # runner default — uses this for fit
    eval_split:  str = "test"                           # runner default — uses this for eval
    models: ModelsConfig = ...
    per_trial_seconds: int = 600
```

Convention: runner reads `train_split` / `eval_split` name pointers (defaults match today's shape). If your splits dict uses different names (e.g. CV folds), override the pointers.

**Pushdown construction lives with the reader, not on `Splits`.** `Splits.to_pyarrow_filter()` was rejected at design time — Splits is format-agnostic; pyarrow is one format. A private helper `_pyarrow_filter_for_ranges(ranges, split_id_col)` lives in `data/registry.py` next to the parquet read calls. If a future storage backend (CSV, other parquet readers, columnar databases) is added, its filter builder co-locates with its reader.

### Q9 — TrialDataContract: rename, generalize, audit-trim

Today's `RunDataContract` (six nested types) becomes four types — renamed away from "Run"/"Snapshot" vocab and generalized to any number of named slices.

```python
@dataclass(frozen=True)
class TrialRef:
    project_name:   str
    experiment_id:  str
    trial_id:       str                          # human id: <number>_<slug> / run_name / trial.id tag
    run_id:         str                          # MLflow run UUID

@dataclass(frozen=True)
class DatasetRef:
    id:             str
    manifest_uri:   str
    identity_hash:  str
    target_column:  str
    split_id_col:   str                          # moved here from old SplitContract
    n_rows:         int
    n_columns:      int

@dataclass(frozen=True)
class SliceContract:
    name:         str | None                     # None for explicit-range loads
    ranges:       tuple[tuple[int, int], ...]
    n_rows:       int
    content_hash: str                            # hash of the loaded slice's content

@dataclass(frozen=True)
class TrialDataContract:
    trial:           TrialRef
    dataset:         DatasetRef
    splits:          dict[str, tuple[tuple[int, int], ...]]   # RunConfig.splits.ranges at trial run time
    slices:          tuple[SliceContract, ...]                # actually loaded by this trial
    schema_version:  int = 1
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, payload) -> TrialDataContract: ...
    def slice(self, name: str) -> SliceContract | None: ...
```

**Renames:** `RunDataContract` → `TrialDataContract`; `RunRef` → `TrialRef`; `SnapshotRef` → `DatasetRef`. (Per §5: a trial is backed by one MLflow run, but the human `trial_id` string and MLflow `run_id` string are distinct and both load-bearing; see 08 Q4 carry-back.)

**Dropped fields** (audit-trim, no guarantee lost):
- `SnapshotRef.prepare_event_id` — Q4 dropped from Dataset.
- `Shape` + `SplitShape` dataclasses — flattened into `DatasetRef.n_rows`/`n_columns` and `SliceContract.n_rows`.
- `SplitContract.view_hash` — per-slice `SliceContract.content_hash` is stronger.
- `SplitContract.train_content_hash` / `test_content_hash` — generalized into `slices: tuple[SliceContract, ...]`.
- `RunDataContract.to_split_view()` — **dropped wholesale, not "ported"** (the 08-carried open item, enumerated). The legacy method emitted a `split_view` dict whose `split_view_hash` was reconstructed-and-compared for integrity; the new design replaces that mechanism with content-addressed per-slice `SliceContract.content_hash` + the L1–L4 validators, so no `to_split_view()` equivalent on `TrialDataContract` is needed. Its actual callers were never the runner: (1) `data/contract.py::validate_run_data_contract` (the view-hash reconstruction) → subsumed by **L1 `validate_trial_data_contract`** (no dict rebuild — compares typed fields); (2) `data/…::validate_split_view` (L3) → **`verify_loaded_slice`** (per-slice `content_hash`); (3) `inspect/views.py::load_data_snapshot` replay → **`load_dataset_by_trial` + per-slice field access** (already mapped, §11 below at "`automl inspect show-trial`"). The runner builds `TrialDataContract` from `LoadedSlice` directly (sub-spec 08 Q4); it never calls `to_split_view`.

**Generalization:** today's hardcoded train/test (in `SplitShape` and `_ranges_to_dict`) becomes free-form: `splits: dict[str, tuple[tuple[int, int], ...]]` records what was in RunConfig at trial run time; `slices: tuple[SliceContract, ...]` records what the trial actually loaded.

**Three integrity layers preserved** (audited — see §5 below for the full diff):

| Layer | Today | Proposed |
|---|---|---|
| L1: Contract ↔ Dataset manifest | `validate_run_data_contract(contract, manifest)` | `validate_trial_data_contract(contract, dataset)` — same checks against typed `Dataset` |
| L2: Loaded data ↔ Dataset manifest | `validate_data_manifest_v2(manifest, df_data, registry_df)` | `validate_loaded_dataset(loaded, dataset)` — recomputes `dataframe_content_hash`/`schema_hash`/`registry_content_hash` and compares to `Dataset.component_hashes.*` |
| L3: Loaded slice ↔ Contract slice hashes | `validate_split_view(manifest, split_view, df_train, df_test)` | `verify_loaded_slice(loaded, slice_contract)` — recomputes `dataframe_content_hash(loaded.df)` and compares to `SliceContract.content_hash` |
| L4: Contract ↔ trial MLflow tags | `_validate_run_lineage` (run_snapshot.py:129) cross-checks contract values against `data.snapshot_*` / `data.split_view_hash` tags on the trial run to detect tag tampering | `verify_trial_tag_lineage(contract, trial_id)` in `data/contract.py` — cross-checks contract values against the new tag scheme (`data.dataset_id` / `data.identity_hash` / `data.manifest_uri` / per-slice `data.slice.<name>.content_hash`). Called from `load_dataset_by_trial` after reading the contract. |

**L2 runs by default at load time (sub-spec 07 dependency, confirmed).** `load_dataset`, `load_dataset_by_id`, and `load_dataset_by_trial` invoke `validate_loaded_dataset` (L2: loaded↔manifest) **by default** before returning — this is what realizes the "`Dataset.component_hashes.*` cross-checked at load time" integrity guarantee below, and sub-spec 07's recipe-only `split_view` integrity leans on it (07 Q2: a `split_view` EvalDataset carries no content/schema hash of its own, so the content-addressed `of_dataset_id` + this L2 check *are* its integrity). `load_dataset_by_trial` additionally runs L3 (`verify_loaded_slice`) + L4 (`verify_trial_tag_lineage`). (A future `strict=False` opt-out is follow-demand; not in v1.)

**Acknowledged drop — `prepare_event_id` format tamper-check.** Today's `validate_run_data_contract` checks that `prepare_event_id` starts with `pe_` and contains the snapshot name (contract.py:251-254). With `prepare_event_id` dropped per Q4 (no consumer beyond logging), this tamper-check is dropped too. The L1 check has one less integrity assertion — accepted: the field was a temporal id only, and the dataset_id / identity_hash cross-checks remain. No replacement integrity assertion added on `created_at` or trial start_time.

**`load_dataset_by_trial` is the audit/reproduction verb.** It reads `TrialDataContract` from MLflow; the contract's `splits` dict (frozen at trial run time) is authoritative for `split_name` resolution. config.py is not read. Six-month-old trials replay exactly.

**MLflow tags** (updated for the rename + per-slice generalization):
```
data.dataset_id               = contract.dataset.id
data.identity_hash            = contract.dataset.identity_hash
data.manifest_uri             = contract.dataset.manifest_uri
data.contract_artifact        = "data_contract.json"
data.slice.<name>.content_hash = slice.content_hash    (one per loaded slice)
data.slice.<name>.n_rows       = slice.n_rows           (one per loaded slice)
```

Replaces today's `data.snapshot_name` / `data.snapshot_identity_hash` / `data.split_view_hash`.

**Writer location:** `mlflow/trial/artifacts/data.py` (per sub-spec 02 §6.3.4 — only the MLflow seam touches MLflow APIs). Runner constructs `TrialDataContract` from `LoadedSlice` objects and calls `mlflow.trial.artifacts.write_trial_data_contract(run_id, payload=contract)`. The runner side is sub-spec 08 territory; this sub-spec specifies the contract API the runner calls.

**Duplicate cleanup:** `data/snapshot.py:506-507`'s `validate_run_data_contract` shim (which just re-imported from `data/contract.py`) is deleted — one canonical home.

---

## 5. Summary — the final shape

### Tier 2 exports (from `automl.data`)

**Types:**
- `Dataset`, `LoadedDataset`, `LoadedSlice`, `DatasetIndex`, `ComponentHashes` (data/dataset.py)
- `DataSpec` (data/spec.py)
- `DataSource` (data/sources/base.py) + concrete `SnowflakeSource`, `LocalCSVSource`, `GCSParquetSource`
- `DataPipeline` (data/pipeline.py)
- `HashKey`, `add_split_id`, `hash_key_columns`, `hash_key_report`, `split_report` (data/split.py — hashing mechanism). **`Splits` is a `project/` export, not data/** — see Q8.
- `FeatureRegistry`, `FeatureEntry` (data/features.py)
- `Profile` (data/profile.py)
- `TrialDataContract`, `TrialRef`, `DatasetRef`, `SliceContract` (data/contract.py)

**Verbs:**
- `build_dataset(*, session=None) → LoadedDataset` — in-memory full pipeline, no persistence
- `materialize(*, refresh_source=False, session=None) → LoadedDataset` — build + persist + register
- `list_datasets(*, session=None) → DatasetIndex`
- `load_dataset(*, split_name=None, split_range=None, session=None) → LoadedSlice | LoadedDataset`
- `load_dataset_by_id(dataset_id, *, split_name=None, split_range=None, session=None) → LoadedSlice | LoadedDataset`
- `load_dataset_by_trial(trial_id, *, split_name=None, split_range=None) → LoadedSlice | LoadedDataset`
- `profile(dataset_id=None, *, session=None) → Profile`
- `get_profile(dataset_id=None, *, session=None) → Profile | None`

**Materialization flow guarantees** (preserved from today's `prepare_data`, ported into the new `materialize()` orchestrator):
- **Source artifact logging.** Each source's `artifact_files(pipeline)` method returns source-specific trace files (e.g., Snowflake's `base_data.executed.sql`, `training_data.executed.sql`). `materialize()` logs these as MLflow artifacts under `data/datasets/<dataset_id>/source_trace/` on the project-overview run. Provenance trail preserved (independent of the dropped `source_event` field on Dataset — that field was runtime metadata; this is the verbatim SQL text).
- **Partial-snapshot guard.** When some but not all GCS objects exist for a candidate dataset_id, `materialize()` refuses to overwrite and raises `StorageError` with the present/missing breakdown (today's `pipeline.py:933-940` behavior, preserved).
- **Idempotent re-materialize.** When all GCS objects exist for the candidate identity_hash, `materialize()` reads the existing manifest, cross-checks it against the candidate (target_column / split_id_col / hash_key / shape — same 9 fields as today's `_validate_existing_snapshot_matches_candidate`), and returns the existing `LoadedDataset` without re-writing.

**Validators (all in `data/contract.py`):**
- `validate_trial_data_contract(contract: TrialDataContract, dataset: Dataset) → None` — L1: contract ↔ Dataset manifest
- `validate_loaded_dataset(loaded: LoadedDataset, dataset: Dataset) → None` — L2: loaded data ↔ Dataset manifest
- `verify_loaded_slice(loaded: LoadedSlice, slice_contract: SliceContract) → None` — L3: loaded slice ↔ contract slice hash
- `verify_trial_tag_lineage(contract: TrialDataContract, trial_id: str) → None` — L4: contract ↔ trial MLflow tags (called by `load_dataset_by_trial`)

**Shared hash primitives** — promoted to **`utils/hashing.py` as PUBLIC functions** per structural spec §13.8 + §10 + appendix (line 827). This is the structural spec's deliberate fix for the cross-domain-privates smell (today `eval/snapshot.py` reaches into `data/snapshot.py`'s underscore-prefixed `_json_hash`):
- `json_hash(value: Any) → str` — deterministic JSON-serializable hash; sha256 prefix (was `_json_hash`)
- `dataframe_content_hash(df: pd.DataFrame) → str` — hash of columns + dtypes + per-row hashes
- `schema_hash(df: pd.DataFrame) → str` — hash of columns + dtypes only

Both `data/` (for `materialize` identity composition + L2/L3 validators) and `eval/` import these from `utils.hashing` as public functions. No domain reaches into another domain's privates. `registry_content_hash` (data-specific — hashes the FeatureRegistry CSV) stays in `data/` since it encodes a data-domain concept, not an AutoML-agnostic primitive.

### Tier 3 extension anchors

- `DataSource` (`data/sources/base.py`) — source-side variation
- `DataPipeline` (`data/pipeline.py`) — orchestration-side variation; 13 documented override hooks

### Private (not exported)

- `_build_pipeline(session, *, refresh_source) → DataPipeline` — internal factory used by the verbs
- `_pyarrow_filter_for_ranges(ranges, split_id_col)` — pushdown construction; reader-co-located
- `_STATS_CHECKS`, `_CHARTS` lists in `data/profile.py`
- Helpers in `data/contract.py`: `_required_dict`, `_required_str`, etc.

### MLflow-layer touchpoints (paths match sub-spec 02 §4 — corrected from earlier draft)

```
mlflow/project/
└── artifacts.py            ← project-scoped writers (per sub-spec 02 §4: dataset_index, data_card, data_learning)
                              EXPANDS to host: profile (write_profile / read_profile),
                              dataset manifest writer (write_dataset / read_dataset),
                              dataset index (read_dataset_index / write_dataset_index),
                              dataset-id resolution (resolve_dataset_id, assign_dataset_id)

mlflow/experiment/
└── datasets.py             ← renamed from snapshots.py per Q1.
                              list_datasets_for_experiment, get_active_dataset, set_active_dataset,
                              resolve_dataset_id (was list_snapshots, resolve_snapshot)

mlflow/trial/artifacts/
└── data.py                 ← already houses contract writer per sub-spec 02 §6.3.4.
                              Renames write_data_contract → write_trial_data_contract;
                              payload type DataContract → TrialDataContract per Q9.
```

Domain code calls these via `automl.mlflow.<noun>.<verb>()` (allowed direction per §9.1). The MLflow API itself only appears inside `mlflow/`.

**Dataset-id resolution algorithm** (was `mlflow/store.py:resolve_snapshot_name` + `SnapshotNameResolution` today): lives in `mlflow/project/artifacts.py` as `resolve_dataset_id(identity_hash: str) → tuple[str, bool]` returning `(dataset_id, was_new)`. Algorithm: read `dataset_index.json`; if `identity_hash` already present → return its existing `dataset_id` + `False`; else bump the version counter (highest `v<n>` + 1) → mint `v<n+1>_<hash8>` → return + `True`. The `v<n>_<hash8>` id format is preserved unchanged from today.

### Integrity guarantees (forward)

| Guarantee | Mechanism |
|---|---|
| A trial identifies which Dataset it used | `TrialDataContract.dataset.id` + `identity_hash` |
| Loaded data matches Dataset manifest (no silent corruption) | `Dataset.component_hashes.*` cross-checked at load time |
| Loaded slice matches contract slice hash | `SliceContract.content_hash` cross-checked at load time |
| Trial replay six months later despite config.py drift | `TrialDataContract.splits` frozen at trial run time |
| Cross-trial comparison is comparable-or-detectably-different | `SliceContract.content_hash` equality check |
| No silent leakage between train and test | split-at-load via `load_dataset(split_name=...)`; model code receives only the slice it asked for |
| Two universes (real / dry_run) strictly isolated | sub-spec 03 path/MLflow routing; Dataset doesn't carry `run_mode` (it's implicit in location) |

---

## 6. Carry-backs to parent specs

**Structural spec (`00-structural-design.md`):**

- §5 Vocabulary: drift-fix list — explicit "snapshot" retirement done at code level (class names, file names, GCS paths, MLflow tags). User-facing noun is Dataset everywhere.
- §5.1 MLflow level table: profile artifacts live under project-overview run (`<project>/overview`), not experiment-overview. Today's behavior is a layering bug fixed by Q5.
- §7 Folder shape: replace `data/profile/{core, publish}.py` with single `data/profile.py`. (Both names rejected: "core" abstract; "publish" empty once MLflow writing moves to `mlflow/artifacts/profile.py`.)
- §7 Folder shape: `data/` includes new `dataset.py` (Dataset / LoadedDataset / LoadedSlice / DatasetIndex / ComponentHashes) and `registry.py` (list/load verbs).
- §7 Folder shape: `data/adapters/` deleted entirely.
- §8.2 `data/` section: Tier 2 exports list expands to include `Dataset, LoadedDataset, LoadedSlice, DatasetIndex, DataPipeline, build_dataset, load_dataset_by_id, load_dataset_by_trial, get_profile, TrialDataContract, TrialRef, DatasetRef, SliceContract`. (`Splits` is a `project/` export per Q8, not data/.) Tier 3 anchor line acknowledges **two** anchors: `DataSource` (sources/base.py) for source-side; `DataPipeline` (pipeline.py) for orchestration-side.

**Sub-spec 01 (`01-project-context.md`):**

- `RunConfig.split: Split` → `RunConfig.splits: Splits` (free-form named dict); the `Split` type is renamed to `Splits` and stays a `project/` export (lives in `project/run_config.py`).
- Add `RunConfig.train_split: str = "train"` + `RunConfig.eval_split: str = "test"` name pointers.
- No new `Session.data_spec` API needed — `_build_pipeline(session)` uses the existing `session.config.require_data_spec()` / `require_run_config()` accessors (sub-spec 01 line 85/87).
- DataPipeline subclass resolution note (sub-spec 01 line 586) updated: resolved by `DataSpec.pipeline_cls` (Q2/Q7), no special `ProjectConfig.load` wiring.

**Sub-spec 02 (`02-mlflow-seam.md`) — folder + signature reconciliation:**

- §4 folder layout: `mlflow/experiment/snapshots.py` → **renamed** `mlflow/experiment/datasets.py`. File rename driven by Q1's "total vocab retirement at code level."
- §4 folder layout: `mlflow/project/artifacts.py` (already lists `dataset_index`, `data_card`, `data_learning`) — annotate that profile writers (`write_profile` / `read_profile`) and the dataset manifest + dataset-id resolution functions also live here.
- §6.2.3 function signatures: rename `list_snapshots(experiment_id=None) → list[SnapshotRef]` to `list_datasets_for_experiment(experiment_id=None) → list[DatasetRef]`; rename `resolve_snapshot(snapshot_hash8) → str | None` to `resolve_dataset_id(identity_hash) → str | None`. The `get_active_dataset` / `set_active_dataset` names stay (already correct); fix the argument name `snapshot_name: str` → `dataset_id: str` in `set_active_dataset`.
- §6.2.3 reference to `SnapshotRef lives in data/dataset.py` → updated to `DatasetRef lives in data/contract.py` (Q9 — DatasetRef is the contract-side reference type).
- §6.3.4 `write_data_contract` / `load_data_contract` → renamed `write_trial_data_contract` / `load_trial_data_contract`; payload type `DataContract` (which referenced `data/contract.py`) → `TrialDataContract`.
- Add `mlflow/project/artifacts.py` writer/loader signatures for profile + dataset + dataset_index + resolve_dataset_id (this is a new section in §6.1).
- Tag-key constants in `mlflow/tags.py`: rename `data.snapshot_*` → `data.dataset_*` / `data.identity_hash`; add per-slice tag scheme `data.slice.<name>.{content_hash,n_rows}`; remove `data.split_view_hash`.

**Sub-spec 03 (`03-cleanup.md`):**

- No changes; `dry_run` strict-isolation principle is preserved by Dataset not carrying `run_mode` (mode implicit in routing).

**Sub-spec 04 (`04-validate.md`):**

- No changes; sub-spec 05's profile-check list pattern + per-check exception wrapping matches sub-spec 04's `_safe` pattern (consistency).

**Sub-spec 07 (eval domain, future) — recorded inbound concerns:**

- **Eval column pre-flight gate.** Today's `pipeline.py:_validate_evaluation_columns` (line 703-718) runs at both materialize-time and load-time to catch mismatches between the active `EvalSpec` and the loaded data (e.g., target column missing from snapshot). This cross-domain check has no forward home in sub-spec 05 — the data domain shouldn't import from eval. Sub-spec 07 needs to specify either: (a) an eval-side validator that runs after `materialize()` / `load_dataset()` via the runner, or (b) a hook the runner calls explicitly between data load and fit. Recording here so sub-spec 07 picks it up; status quo behavior is preserved by the runner sequencing both `materialize()` then `eval_spec.validate_columns(loaded.df, target)` explicitly.
- **Eval domain imports of hash primitives.** `eval/snapshot.py` imports `_json_hash`, `dataframe_content_hash`, `schema_hash` from `data/snapshot.py` today. After sub-spec 05's `data/snapshot.py` deletion these move to `data/dataset.py` (see §5 "Shared hash primitives" above). Sub-spec 07 imports stay valid; only the import path string changes.

**CLI (structural spec §11.1 — already specifies these; recording alignment):**

- `automl data profile [<dataset>]` — structural spec §11.1 (line 459) already specifies this verb mapping to `data.profile`, **replacing the top-level `automl profile`** (§11.1 line 492). Implementation calls `data.profile.profile()` (the new Q5 verb). No new carry-back needed — 05's verb matches the catalog.
- `automl data materialize` — structural spec §11.1 (line 473) already lists this ("runner-driven via Python today; no skill/CLI use case yet"). It is the replacement for today's `python -m automl.data.prepare` entry point. `data/prepare.py` is deleted; skill scripts that ran `python -m automl.data.prepare` migrate to `uv run automl data materialize`.
- `automl data list` — structural spec §11.1 (line 458) maps to `data.registry.list_datasets`. Matches 05's `list_datasets` verb.
- `automl inspect show-trial` and similar verbs whose Python implementation uses `inspect/views.py:load_data_snapshot(run_id)` (today returns a single object with `df_train` + `df_test` + `split_view` + `data_contract`) migrate to: `load_dataset_by_trial(trial_id)` for the dataframe + `mlflow.trial.artifacts.load_trial_data_contract(trial_id)` for the contract. Two calls replace one; inspect sub-spec captures the surface adjustment.

---

## 7. Open items (recorded for closeout / future sub-specs)

- **YAML / SQL-comment metadata loader for projects with many features.** Description + tags rejected this round (no source of data). If a future project genuinely needs them, add an opt-in `projects/<name>/feature_metadata.yaml` loader. Deferred per `feedback_extension_points_follow_demand`.
- **Per-feature validation rules subsystem** (min/max, allowed values, NOT NULL). Real value but separate concern; defer.
- **Cross-trial feature importance aggregation** for proposer context. Belongs in `experiment/views/` (sub-spec 09), not on FeatureRegistry.
- **Drift detection across Datasets.** Belongs in `experiment/views/diagnostics.py` (structural spec §15 deferred placeholder).
- **Physical parquet partitioning by SPLITID** for true skip-the-bytes pushdown. Today's single-file parquet supports filter pushdown via row-group pruning only — **and that only works if rows are written in SPLITID sort order** (otherwise row groups span all splits and pruning is impossible). **Aspirational guarantee for now:** `materialize()` is permitted but not required to sort rows by `split_id_col` before writing parquet. If row-group profiling shows pushdown is ineffective, two options emerge in priority order: (1) add `df.sort_values(split_id_col)` to `materialize()` (cheap; preserves single-file format); (2) physically partition the parquet directory by SPLITID range (`data/datasets/<id>/SPLITID=0-19/`, `SPLITID=20-39/`, …). Decision deferred to implementation time when there's measured cost.
- **Project-side custom profile checks** (`projects/<name>/profile_checks.py` parallel to `projects/<name>/validators.py`). Same defer-until-real-demand pattern as project-side validate extensions.
- **Project-level "learning" subsystem** (golden/weak feature artifacts, `apply_learning_flags`). Deferred per README "Out of scope"; sub-spec 06 / 07 / 09 do NOT migrate it. Future redesign.

---

## 8. Sub-spec status

- **Approved:** 2026-05-24 (after three-agent review + holistic cross-doc consistency pass).
- **Three reversals applied** (structural spec won over sub-spec draft): hash primitives → `utils/hashing.py` public (§13.8); CLI verb `automl data profile` (§11.1); `Splits` → `project/run_config.py` (dependency-cycle avoidance).
- **Carry-backs APPLIED** to 00 (§5/§5.1/§7/§8.1/§8.2/§13.8/§719/appendix), 01 (import line + DataPipeline subclass note), 02 (§4/§6.1/§6.2.3/§6.3.4/§9). None pending.
- **Pre-existing 00↔02 mlflow folder-layout inconsistency** fixed in the same pass (00 now describes per-noun folders per 02 §4).
- **Migration checklist updated:** data-domain `[?]`/`[/]` rows resolved; drops marked `[-]`; renames recorded.
- **open-questions.md updated:** sub-spec 05 items logged; two items carried to sub-spec 07 (eval pre-flight gate, `of_data_snapshot_id` naming); Dataset/EvalDataset unification stays open through 07.

---

*End of sub-spec 05.*
