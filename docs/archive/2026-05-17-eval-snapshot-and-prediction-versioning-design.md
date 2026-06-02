# Eval snapshot and prediction versioning — design

**Status:** draft, awaiting user approval before plan
**Authors:** SoulEvill + Claude
**Date:** 2026-05-17

## Motivation

Today's evaluation logging has three pain points:

1. **MLflow holds parquet files.** `eval/predictions.parquet` is logged as an
   MLflow artifact (and additionally duplicated to GCS), inflating MLflow's
   store with row-scale data it isn't designed to hold.
2. **No eval-data identity.** Unlike data snapshots — which are
   content-addressed, deduplicated, and stable across trials — eval data has
   no identity. Re-evaluation today (`automl.reevaluate`) takes a free-form
   `df_eval` plus a string `label`, logs a child MLflow run, and provides no
   way to know whether two re-evals scored against "the same data."
3. **Predictions don't join back.** Predictions are stored per-trial-run with
   only positional row indices. There's no externally meaningful key to join a
   model's predictions back to the training data, nor to compare five models'
   predictions against one another.

A related issue: training-set metrics aren't logged, so overfit detection is
manual.

This spec consolidates a refactor that resolves all three concerns
simultaneously, because they share the same root cause: eval data lacks the
content-addressed identity that data snapshots already have.

## Non-goals

- Backward compatibility with existing MLflow runs. The cutover is hard;
  old runs stay readable in MLflow but new code does not try to reconcile
  their shapes.
- Garbage collection of orphan predictions or augmentations (later
  concern; content-addressed storage grows monotonically).
- Auto-pulling augmentation data from external systems. The user owns the
  loader; the framework owns governance.
- Multi-target / multi-output eval. `target_column` is single-string.
- A "diff two eval snapshots" inspect view (useful but separate).
- High-level prediction-join helpers (`load_predictions(eval_snapshot_id,
  model_run_ids=…)`, `split_view_id(snap, partition=…)`). The
  underlying GCS layout makes these trivially implementable as a follow-up
  (see "Open follow-up: prediction-join helpers"). Data snapshots stay
  strict (no augmentation primitive on the data side).

## Glossary

- **trial run** / **model run** — used interchangeably. The MLflow run that
  owns a trained model artifact. Its MLflow `run_id` is the canonical
  identifier (`model_run_id` and `trial_run_id` refer to the same string).
- **eval snapshot** — content-addressed labeled frame identified by
  `eval_snapshot_id = v<version>_<hash8>`.
- **hash_key** — the project-declared business identifier columns. **One
  unified concept** — same field used for SPLITID derivation and for
  per-row identity in eval snapshots, predictions, and augmentations.
  Accepts a single column name (`"SK_ID_CURR"`) or a list of column names
  (`["customer_id", "as_of_date"]`). **Must be unique per row** —
  validated at pipeline init. Distinct from `split_id_col` (a 0–99 bucket,
  not unique).

## Core concept — two ideas, kept separate

Today's `snapshot_identity_hash` conflates two things. This spec keeps them
distinct:

| Concept | What it is | What it enables |
|---|---|---|
| **Eval-snapshot identity** | Content hash of an eval frame | "Is this the same eval data?" Provable, immutable. |
| **Model–eval compatibility predicate** | Boolean over `model.required_input_columns ⊆ eval.columns` (with dtype match) | "Can model M score eval snapshot E?" Decoupled from identity. |

This split dissolves the "added a column → identity changed" pain without
introducing a fuzzy notion of "almost the same eval."

## Eval snapshot identity

Every eval frame becomes an immutable, content-addressed artifact named
`v<version>_<hash8>`, symmetric to data snapshots.

Two kinds:

- **`split_view`** — pointer only, no GCS data file. Manifest declares
  `{kind: split_view, of: <data_snapshot_id>, split_id_col, buckets}`. The
  `eval_snapshot_id` is deterministic from `(data_snapshot_id, split_id_col,
  sorted(buckets))`. Trial-time test view, train view, retrospective slices
  are all this kind. **Zero GCS bytes added for the common case.**
- **`external`** — labeled frame published to GCS, hashed by content. Used
  for OOT data, vintage backtests, anything not derivable from an existing
  data snapshot.

Identity hash components:

```
kind ∈ {split_view, external}
target_column
hash_key (sorted list of column names — single-column projects use [col])

If kind == split_view:
    data_snapshot_id, split_id_col, sorted(buckets)
If kind == external:
    content_hash_of_frame, schema_hash
```

`hash_key` is the **load-bearing invariant**. Every project declares it
in `DATA.hash_key`. Every eval snapshot publishes with it (either inherited
from the source data snapshot or carried on the publishing call). It
**must be unique per row** — the pipeline validates this at materialization
and refuses to publish a non-unique snapshot. Without uniqueness, the
prediction-join story and the augmentation-join story both die.

Composite `hash_key` (a list of columns) is fully supported. Predictions,
augmentations, and joins use all of the hash_key columns. Uniqueness is
on the tuple of values.

There is **no separate `row_id_col` concept** — the prior split between
"hash for splitting" and "id for joining" is gone. One field, one
contract.

### Eval snapshot `manifest.json` schema

Symmetric to the data snapshot manifest. Validators on the read path
verify every field against the loaded frame (or against the referenced
data snapshot, for split views).

```json
{
  "schema_version": 1,
  "project_name": "example_homecredit",
  "experiment_id": "example-homecredit",
  "eval_snapshot_id": "v1_e8a4c102",
  "kind": "split_view",
  "target_column": "TARGET",
  "hash_key": ["SK_ID_CURR"],            // always a sorted list, even for single-column
  "created_at": "2026-05-17T10:32:11Z",
  "shape": {"n_rows": 61504, "n_columns": 122},

  "hashes": {
    "eval_snapshot_hash": "sha256:...",
    "schema_hash": "sha256:...",
    "content_hash": "sha256:..."
  },

  // present only when kind == split_view
  "split_view": {
    "of_data_snapshot_id": "v3_a1b2c3d4",
    "split_id_col": "SPLITID",
    "buckets": [[80, 100]]
  },

  // present only when kind == external
  "gcs": {
    "data_uri": "gs://.../eval/snapshots/v1_e8a4c102/data.parquet",
    "manifest_uri": "gs://.../eval/snapshots/v1_e8a4c102/manifest.json"
  },

  "provenance": {                       // free-form, caller-supplied
    "vintage": "2026Q2",
    "source": "warehouse table X"
  }
}
```

Augmentation `manifest.json` schema:

```json
{
  "schema_version": 1,
  "eval_snapshot_id": "v1_e8a4c102",
  "name": "ltv",
  "hash8": "a3f1c204",
  "hash_key": ["SK_ID_CURR"],                   // matches the eval snapshot's hash_key
  "columns": [{"name": "LTV", "dtype": "float64"}],
  "shape": {"n_rows": 60012, "n_columns": 1},   // n_columns excludes hash_key columns
  "content_hash": "sha256:...",
  "created_at": "2026-05-17T11:04:22Z",
  "source": {                                    // free-form, caller-supplied
    "sql_path": "eval_sql/ltv.sql",
    "as_of": "2026-Q2",
    "definition": "loan_amount / property_value at scoring time"
  }
}
```

## GCS layout

```
<gcs_root>/<route>/
  data/snapshots/v<n>_<hash8>/         # unchanged
    data.parquet
    feature_registry.csv
    manifest.json
  eval/snapshots/v<n>_<hash8>/         # NEW
    manifest.json                      # always present
    data.parquet                       # only if kind == external
    augmentations/<name>__<hash8>/
      data.parquet                     # (*hash_key, added_cols...)
      manifest.json
  predictions/<eval_snapshot_id>/      # NEW — keyed by eval snapshot
    <trial_run_id>.parquet             # (*hash_key, y_pred, y_proba_*)
    <trial_run_id>.json                # manifest metadata (see below)
  mlflow/<run_id>/                     # unchanged shape; smaller content
```

`route` here is the same as today: `route_prefix_for(...)` from
`automl/mlflow/artifacts/gcs_paths.py`. **Dry-run eval snapshots route the
same way data snapshots do today** — under
`<gcs_root>/dry_run/<project>/<experiment_id>/eval/...`. No new behavior.

Predictions live under `predictions/<eval_snapshot_id>/` (not under
`runs/<run_id>/`). This makes "join N models' predictions on one eval" a
glob over the eval directory.

`y_true` is **never** copied into a prediction file. It lives in the eval
snapshot once.

**Prediction manifest (`<trial_run_id>.json`):** a tiny JSON next to the
parquet that makes a loose parquet self-describing without opening it.
Without this, someone holding a copy of the parquet has no idea what it
is.

```json
{
  "schema_version": 1,
  "trial_run_id": "abc123...",
  "eval_snapshot_id": "v1_e8a4c102",
  "eval_snapshot_kind": "split_view",
  "label": "test",
  "hash_key": ["SK_ID_CURR"],
  "row_count": 61504,
  "augmentations_used": [{"name": "ltv", "hash8": "a3f1c204"}],
  "written_at": "2026-05-17T10:32:11Z"
}
```

The label is included so the file is recognizable by its purpose ("test",
"train", "oot_q2_2026") rather than just its hash.

## Augmentations (additive-only manifests)

Lets a user enrich an existing eval snapshot with new columns (LTV,
TARGET_AS_OF_2026Q2, etc.) without invalidating prior predictions.

Rules enforced at publish:

1. All columns in the eval snapshot's `hash_key` must be present in the
   augmentation frame, and the tuple of `hash_key` values must be unique.
2. Augmentation columns must **not** overlap any column in the base eval
   snapshot. (Override is forbidden.)
3. Augmentation columns must not overlap a column already published by
   another augmentation on the same eval snapshot.
4. `set(aug[hash_key tuples]) ⊆ set(eval[hash_key tuples])`. Orphan rows
   refused.
5. Coverage gaps are allowed (an aug may cover a subset of eval rows).
6. Dtypes must be parquet-serializable.
7. `name` matches `^[a-z][a-z0-9_]*$`.
8. Identical content (same `eval_snapshot_id` + `name` + `content_hash`) →
   no-op idempotent publish.

If LTV's definition changes, the user publishes a new augmentation; new
content → new `aug_id` (`ltv__<new hash8>/`). Old metrics tied to the old
hash remain computable; new metrics tied to the new hash get their own
record. **Content addressing makes "break" impossible — only "diverge with
a new name."**

Metrics declare `required_augmentations: tuple[str, ...]`. At evaluate
time, the runner left-joins the requested augmentations on the eval
snapshot's `hash_key` columns before calling `Metric.compute`.

## Public API surface

### Eval snapshot publishing

```python
from automl.eval import prepare_eval_snapshot, prepare_eval_split_view

# External: labeled frame from anywhere.
# hash_key accepts a string (single column) or list (composite). Uniqueness
# of (hash_key tuples) is validated at publish — non-unique raises.
eval_snap = prepare_eval_snapshot(
    frame=df_oot,
    target_col="TARGET",
    hash_key="SK_ID_CURR",
    provenance={"vintage": "2026Q2", "source": "warehouse table X"},
)

# Split view: derived from a data snapshot. hash_key is inherited from the
# referenced data snapshot — caller does not pass it here.
eval_snap = prepare_eval_split_view(
    data_snapshot_id="<data snapshot id>",
    split_id_col="SPLITID",
    buckets=[(80, 100)],
)
```

### Augmentation publishing

```python
from automl.eval import prepare_eval_augmentation

aug = prepare_eval_augmentation(
    eval_snapshot_id="v1_e8a4c102",
    frame=ltv_df,
    name="ltv",
    source={"sql_path": "eval_sql/ltv.sql", "as_of": "2026-Q2"},
)
```

### Evaluate (the single verb)

Lives at `automl.eval.evaluate`. CLI verb: `automl eval`.

```python
from automl.eval import evaluate

result = evaluate(
    model_run_id="abc123",
    eval_snapshot_id="v1_e8a4c102",
    eval_spec=None,                  # default: project's CURRENT EvalSpec
    label=None,                      # default: "eval_<eval_snapshot_id_short>"
    overwrite=False,                 # default: insert-update (upsert)
    set_as_primary_label=False,      # if True, this label becomes the trial run's primary_label
)
# EvaluateResult(
#   trial_run_id, eval_snapshot_id, label,
#   predictions_uri, metrics={...}, primary_metric="auc",
#   is_primary_label=True/False,
#   cached=True/False, mlflow_url=…
# )
```

`label` is the human-readable handle stored on the trial run's eval tree
(see "Trial run artifact tree"). Trial-time runner passes `"train"` and
`"test"`. Re-eval defaults to `f"eval_{short_hash}"`; users can override.

`eval_spec=None` resolves to the project's **current** `EvalSpec` at
re-eval time. The trial run's frozen `report.json` always records which
metrics were computed (so historical reproducibility lives in the report,
not in re-running with an old spec). This is what makes "add a new
metric to project.py and re-run evaluate against an old trial" cheap.

`set_as_primary_label=True` is how the leaderboard primary moves — it's
triggered through the eval process itself, not a separate setter. The
trial-time runner calls `evaluate(...)` with `set_as_primary_label=True`
on its `"test"` call. Any later re-eval can opt into becoming the new
primary the same way. No `automl.set_primary_label` helper — primary
management is part of the upsert flow.

### Loading helper (only the foundational one)

```python
eval_snap = automl.load_eval_snapshot(eval_snapshot_id)
# For a split_view eval snapshot, this rehydrates the frame from the
# referenced data snapshot on demand (no GCS data file exists for the
# eval snapshot itself).
```

Higher-level prediction-join helpers (`load_predictions`, `split_view_id`,
etc.) are deferred — see "Open follow-up — high-level prediction-join
helpers" at the end. The GCS layout already makes
them trivially implementable; we want the foundation solid before deciding
ergonomics.

## Idempotency semantics (upsert, not overwrite-all)

`evaluate(...)` performs an **insert-update** at the per-metric level. New
metric names always append. Existing metric values are only touched when
`overwrite=True`. The flag never wipes the metric list — it only governs
whether existing values can be replaced.

| Situation | Behavior |
|---|---|
| `(model_run_id, eval_snapshot_id, label)` predictions in GCS AND eval report in MLflow with matching metrics | No-op. Return cached. |
| Predictions exist, report missing | Compute requested metrics from cached predictions; write report. No re-prediction. |
| Neither exists | Score → write predictions → run EvalSpec → write report. |
| Predictions/report exist but `overwrite=True` | Re-score, replace predictions and report in-place. |
| New `eval_spec` declares **new** metric names | Reuse predictions; compute only the new metrics; **append** to the existing report's `metrics` list. |
| New `eval_spec` declares **already-logged** metric names, `overwrite=False` | Reuse cached values for those names; do not refuse. (Pure cache hit.) |
| New `eval_spec` declares **already-logged** metric names, `overwrite=True` | Recompute and replace just those metric entries. Other entries untouched. |
| New `eval_spec` retargets `primary` | **Always free** — no metric recompute. Pointer updates on the report and the MLflow tag; unprefixed scalar metric is re-aliased to the new primary's value. |
| `(label, eval_snapshot_id)` already exists with a **different** `eval_snapshot_id` for the same label | Hard error: "label `<label>` already maps to `<existing_id>`; pass a different label or `overwrite=True` to replace." |

"Metric name" means the **resolved name** (`Metric.resolved_name()`, the
alias-or-class-derived string already used by `EvalSpec` today). Two
metrics that resolve to the same name collide; aliased metrics are
distinct.

**Never** writes a child MLflow run. Eval lives on the trial run, indexed
by label. Same trial run can accumulate multiple `eval/<label>/report.json`
entries over its lifetime — train, test, and any number of later re-evals
under user-named labels.

**Train-eval failure handling:** when the trial-time runner calls
`evaluate(...)` for both `train` and `test`, train is **best-effort**.
A failure during train scoring is logged as a warning and the trial
continues; train metrics are simply absent. A failure during test
scoring is trial-fatal (same as today's eval failure path). This matches
how training-set metrics are an observability win, not a correctness
contract.

## `EvalSpec` redesign — unified metrics list, primary is a pointer in storage

Today's `EvalSpec(primary=Auc(), metrics=[...])` gives `primary` special
status in **storage**: different path in the report, different log path
in MLflow. That's what creates the "rename primary = restructure
storage" problem.

The fix is to keep the **constructor shape that's already familiar** —
`primary` is one slot, `metrics` is "the other metrics" — but unify
storage so primary is just a pointer:

```python
EvalSpec(
    metrics=[LogLoss(), KSStatistic()],   # the other metrics
    primary=Auc(),                         # the primary metric instance
)
```

Constructor rules:

- `primary` is a `Metric` instance (required).
- `metrics` is a sequence of `Metric` instances (default empty) — the
  *other* metrics, distinct from `primary`.
- Internally: `all_metrics = (primary, *metrics)` becomes one ordered list.
  `primary_name = primary.resolved_name()`.
- All resolved names across `(primary, *metrics)` must be unique. Hard
  error on collision.
- `metrics=[]` is valid — `EvalSpec(primary=Auc())` declares a single
  primary metric and nothing else.

This keeps a clean, type-consistent constructor (everything is a `Metric`,
no string names at definition time) and avoids declaring the primary
twice.

What this changes downstream:

- `report.json` stores every metric the same way under `metrics:`;
  `primary` is a top-level **string pointer** that names which entry is
  the primary.
- MLflow scalar metrics: every metric is logged at `<label>.<metric_name>`.
  The pointed-to primary is **additionally** logged unprefixed under
  `<metric_name>` (for leaderboard sorting). When the pointer moves, the
  unprefixed metric is re-logged with the new primary's value.
- Re-evaluating with a new `EvalSpec` that only changes the primary (same
  set of metric resolved names, different primary) is a pure pointer
  update. Zero recomputation. Zero data movement.

This collapses the "primary vs others" split in storage while keeping the
familiar declarative constructor shape.

## Model–eval compatibility predicate

Before scoring, `automl.eval.evaluate` verifies:

```python
required = set(model.required_input_columns)   # from features/model_feature_registry.csv
available = set(eval_snapshot.columns)
if not required.issubset(available):
    raise ColumnMissing(missing=required - available)

required_dtypes = model.required_input_dtypes()  # (name, dtype)
for name, dtype in required_dtypes.items():
    if eval_snapshot.dtypes[name] != dtype:
        raise DtypeMismatch(name=name, expected=dtype, got=…)
```

No silent coercion. Fail with a clear, fixable error.

## Trial-time integration

`automl/runner/_execute.py` changes:

1. After model trains, derive two split-view eval snapshots:
   ```python
   test_eval  = prepare_eval_split_view(..., buckets=test_buckets)
   train_eval = prepare_eval_split_view(..., buckets=train_buckets)
   ```
2. Score both via `evaluate(...)`, with stable human-readable labels:
   ```python
   evaluate(model_run_id, test_eval.id,  label="test",
            set_as_primary_label=True)   # test is the trial-time primary
   try:
       evaluate(model_run_id, train_eval.id, label="train")
   except Exception as e:
       log.warning("train-eval failed (best-effort): %s", e)
   ```
   Each successful call writes its own predictions to GCS and its own
   report under `eval/<label>/report.json`.
3. The `eval/manifest.json` is written/updated by `evaluate(...)`.
4. MLflow scalar metrics: every metric logged as `<label>.<metric>` (e.g.
   `test.auc`, `train.auc`). The `primary_label`'s primary metric is
   additionally logged unprefixed for leaderboard sorting (e.g.
   `auc = 0.7821`). The existing MLflow tag `eval.primary_metric` is
   preserved (set to the resolved primary metric name).

The old `write_predictions` / `write_evaluation_results` paths inside
`_execute` are removed entirely. `evaluate(...)` is the only writer.

## Trial run artifact tree (final)

Evals are keyed by **label**, not by hash, so a human reading the artifact
tree immediately knows what's what. The hash lives inside each
`report.json` as `eval_snapshot_id`. An `eval/manifest.json` at the top of
the eval tree acts as a table of contents.

```
mlflow/<trial_run_id>/artifacts/
  model.pkl
  features/
    feature_importance.csv
    model_feature_registry.csv
    model_report.json
  data/
    contract.json
  eval/
    manifest.json                  # table of contents (see below)
    train/report.json
    test/report.json
    oot_q2_2026/report.json        # user-named label on re-eval
  validation/report.json
  timing/timing.json
  manifest.json
```

`eval/manifest.json` schema (every label that has a `report.json` under
`eval/` appears here as a row — train, test, and any user-named re-evals):

```json
{
  "schema_version": 1,
  "primary_label": "test",
  "evaluations": [
    {"label": "train",       "eval_snapshot_id": "v1_b7c1...",
     "kind": "split_view",   "primary_metric": "auc",
     "computed_at": "2026-05-17T10:32:11Z"},
    {"label": "test",        "eval_snapshot_id": "v1_e8a4...",
     "kind": "split_view",   "primary_metric": "auc",
     "computed_at": "2026-05-17T10:32:11Z"},
    {"label": "oot_q2_2026", "eval_snapshot_id": "v2_a3f1...",
     "kind": "external",     "primary_metric": "auc",
     "computed_at": "2026-05-18T09:11:04Z"}
  ]
}
```

`primary_label` identifies which entry's primary metric is logged
unprefixed as the leaderboard sort key. Trial-time runner sets this to
`"test"` by calling `evaluate(..., label="test", set_as_primary_label=True)`.
Users move it later by calling `evaluate(..., set_as_primary_label=True)`
on whichever label they want as the new primary. **There is no separate
"set primary label" verb** — primary management is triggered through the
eval process itself.

`report.json` schema (per eval):

```json
{
  "schema_version": 2,
  "label": "test",
  "eval_snapshot_id": "v1_e8a4c102",
  "eval_snapshot_kind": "split_view",
  "predictions_uri": "gs://…/predictions/v1_e8a4c102/<run_id>.parquet",
  "augmentations_used": [{"name": "ltv", "hash8": "a3f1c204"}],
  "primary": "auc",
  "metrics": [
    {"name": "auc",     "value": 0.7821, "augmentations": []},
    {"name": "logloss", "value": 0.4123, "augmentations": []}
  ],
  "computed_at": "2026-05-17T10:32:11Z"
}
```

`primary` is the **pointer**; the primary metric's value is read from the
`metrics` list. No duplicate storage.

Label collision rules at `evaluate(...)`:

- Same `(label, eval_snapshot_id)` → idempotent upsert (rules above).
- Different `eval_snapshot_id` under same `label` → hard error with a
  message: "label `<label>` already maps to `<existing_id>`; pass a
  different label or `overwrite=True` to replace."
- A change to `primary_label` is a pointer move only; the underlying
  per-label entries are untouched.

## Code surface

### New modules

- `automl/eval/snapshot.py` (~250 LOC) — `EvalSnapshotIdentity`, hashing,
  manifest builder + validator, GCS path helpers. Mirrors
  `automl/data/snapshot.py`.
- `automl/eval/publish.py` (~150 LOC) —
  `prepare_eval_snapshot`,
  `prepare_eval_split_view`,
  `prepare_eval_augmentation`.
  All idempotent on content hash.
- `automl/eval/evaluate.py` (~200 LOC) — the `evaluate` verb. Orchestrates
  idempotency, train-eval-as-best-effort, prediction writes,
  `eval/manifest.json` upsert, and `primary_label` pointer moves. Emits
  `EvaluateResult`. Public re-export: `from automl.eval import evaluate`.
- `automl/eval/loading.py` (~50 LOC) — `load_eval_snapshot` only (rehydrate
  split-view or external frames; verify manifest hashes against bytes).
  High-level prediction-join helpers (`load_predictions`) are deferred —
  not in this module yet.
- `automl/mlflow/artifacts/predictions.py` (~100 LOC) — GCS writer keyed by
  `(eval_snapshot_id, trial_run_id)`. Writes parquet
  `(*hash_key, y_pred, y_proba_*)` AND manifest JSON metadata
  `(trial_run_id, eval_snapshot_id, label, hash_key, row_count,
  augmentations_used, written_at)`. Composite `hash_key` produces multiple
  ID columns in the parquet.
- `automl/cli/eval.py` — `automl eval` verb. Thin argparse wrapper around
  `automl.eval.evaluate`. Registered via `@register("eval", ...)` per the
  package's CLI convention.

### Modified modules

- `automl/eval/base.py` — `Metric.required_augmentations: tuple[str, ...] = ()`.
  `EvalSpec` redesign: constructor takes `primary: Metric` and
  `metrics: Sequence[Metric] = ()`. Internally unified into one ordered
  list `(primary, *metrics)`. `metrics=[]` is valid. Storage layer (the
  thing that produces report.json) writes all metrics under one list and
  emits `primary` as a string pointer. Augmentations are left-joined on
  the eval snapshot's `hash_key` columns before calling `Metric.compute`.
- `automl/eval/runner.py` — accepts `eval_snapshot_id` instead of `df_test`;
  loads the frame from the snapshot manifest.
- `automl/runner/_execute.py` — replace the bespoke eval/predictions code
  (~lines 740–830) with two `evaluate(...)` calls (test with
  `set_as_primary_label=True`, train as best-effort).
- `automl/data/pipeline.py` — validate that `DATA.hash_key` (str or list)
  produces unique tuples across the materialized frame; refuse to publish
  otherwise. Record the normalized `hash_key` (always a sorted list of
  column names) on the data snapshot manifest as a top-level field.
- `automl/mlflow/artifacts/eval.py` — `write_evaluation_results` updated to
  write per-label reports at `eval/<label>/report.json` and maintain
  `eval/manifest.json`. `write_predictions` deleted.
- `automl/mlflow/artifacts/__init__.py` — drop `write_predictions` export.
- `automl/inspect/views.py` — leaderboard reads the new metric namespacing
  (primary unprefixed, others prefixed); show-trial reads the per-eval
  report tree via `eval/manifest.json`.
- `automl/cli/__init__.py` — register the new `eval` verb.
- `automl/loop_context/` — proposer/coder context render reads `eval/manifest.json`
  to surface the current `primary_label` and its `primary_metric`. The
  agent does NOT pin to a historical primary; it reads the manifest each
  turn. (Update is mechanical: replace direct reads of the old single
  `eval/report.json` with reads of the per-label tree.)

### Deleted modules / paths

- `automl/reevaluation.py` — replaced by `automl/eval/evaluate.py`.
- The duplicate GCS prediction upload inside `automl/runner/_execute.py`
  (runs/<YYYY-MM>/<run_id>/eval/predictions.parquet path).
- `eval/predictions.parquet` as an MLflow artifact path.
- The top-level `eval/report.json` path (replaced by per-label
  `eval/<label>/report.json` files with `eval/manifest.json` as TOC).

### Tests

- `tests/unit/eval/` — identity hashing, manifest round-trips, publish
  validations (each rule has a dedicated test), augmentation rules.
- `tests/integration/eval/` — end-to-end `evaluate` from snapshot;
  augmentation join; idempotency (cached / appended / overwrite);
  trial-time train+test split-view derivation; predictions reload.
- `tests/contracts/` — pin new GCS layout, new MLflow artifact tree, single
  `evaluate` verb; remove pins for retired paths.
- `tests/regression/` — regenerate golden manifests for the new trial run
  shape.

## Notebook updates

- `6_reevaluate_existing_model.ipynb` — rewritten to use
  `prepare_eval_split_view` + `evaluate`, no `automl.reevaluate`.
- `5_inspect_logged_runs_and_artifacts.ipynb` — surface the new
  per-eval-snapshot report tree and predictions join helper.

## Open questions for the implementation plan

These are decisions deferred to the implementation plan, not unresolved
design questions:

- **Augmentation-name resolution at evaluate time.** If two augmentations
  on the same eval snapshot share a `name` slug but differ by `hash8`, what
  resolves? Proposed default: most-recently-published wins, with a way to
  pin by `hash8` in `Metric.required_augmentations` (`"ltv@a3f1c204"`).
- **Compatibility predicate dtype comparison.** Numpy/pandas dtype objects
  don't compare cleanly across pandas versions; need a canonical
  representation (likely string-normalized).

## Open follow-up — high-level prediction-join helpers

The GCS layout in this spec makes the following helpers trivially
implementable, but their ergonomics aren't decided yet:

- `automl.load_predictions(eval_snapshot_id, model_run_ids=[...])` —
  glob + read + merge predictions on the eval snapshot's `hash_key`
  columns for many models against one eval snapshot. Wide vs long format,
  whether `y_proba_*` columns are included by default, namespacing scheme
  for column names — all TBD.
- `automl.split_view_id(data_snapshot, partition="test")` — convenience
  to get the eval_snapshot_id for the train/test view of a loaded data
  snapshot, without re-publishing.
- `automl.list_evals(model_run_id)` — read the trial run's
  `eval/manifest.json` and summarize.

These are pure conveniences on top of the foundation in this spec.
Deferring them keeps the cutover minimal and lets us decide the API after
real usage shapes the requirements.

## What this gives the user

- Predictions move out of MLflow entirely (your #1 ask).
- Eval data has identity, lives in GCS once, dedup-by-hash (your #2 ask).
- Re-eval is an **upsert**: new metric names append, existing metric values
  only change when `overwrite=True`. Primary metric is a free pointer
  update — moving primary requires zero recomputation (your #3 ask).
- Eval tree is **human-readable by label** (`train/`, `test/`, user-named),
  with an `eval/manifest.json` as the table of contents. The hash lives
  inside each report.
- Predictions join back to data on the project's `hash_key` columns
  natively — the GCS layout makes `glob('predictions/<eval_id>/*.parquet')`
  the join primitive. High-level helpers deferred (see open follow-up
  section).
- Training-set metrics logged automatically via the same primitive (your
  overfit-detection ask). Same code path as test eval.
- "Eval data goes through the same governance gate as training data" — same
  `prepare → publish → reference by id` shape as data snapshots (your
  governance instinct).
- `EvalSpec` keeps the familiar constructor shape (one `primary=` slot,
  `metrics=` for the others) but unifies storage so primary is a pointer.
  Adding/replacing metrics is mechanical; changing primary is a free
  pointer move.
- Project-agnostic: the existing `DATA.hash_key` (str or list) is the
  single source of truth for both splitting and per-row identity. New
  contract: the pipeline validates uniqueness of the hash_key tuple at
  init and refuses to publish a non-unique snapshot. No separate
  `row_id_col` field; no auto-promotion logic. One unified concept across
  data snapshots, eval snapshots, predictions, and augmentations.
