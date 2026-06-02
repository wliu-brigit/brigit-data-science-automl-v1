# Sub-spec 07 — Eval domain

**Status:** APPROVED 2026-05-24 (design interview Q1–Q6 + three-agent review + fixes applied)
**Started:** 2026-05-24
**Parent:** `00-structural-design.md` §8.4 (eval domain), §13.8 (Dataset/EvalDataset seam)
**Depends on:** sub-spec 05 (Data) — uses data's public slice API; sub-spec 02 (MLflow seam) — eval artifacts/predictions writers; sub-spec 04 (Validate) — eval/metric checks.

This sub-spec settles the interface for the `eval/` domain: the `Metric` ABC +
builtins, `EvalSpec`, the `EvalDataset` identity family, the `evaluate` verb,
the eval results/predictions schemas, and the eval-side checks. It also resolves
the three items carried into 07 from earlier sub-specs:

1. 🟡 Dataset / EvalDataset unification checkpoint (open-questions.md, pre-05 alignment).
2. 🟡 Eval column pre-flight gate home (carry-back from 05).
3. 🟡 `of_data_snapshot_id` → `of_dataset_id` naming (carry-back from 05).

---

## Scope (from §00 §8.4)

- **Owns:** `Metric` ABC + builtins (`Auc`, `LogLoss`, `ThresholdSweep`), `EvalSpec`,
  `EvalDataset` (the held-out evaluation view; distinct from training `Dataset`),
  the `evaluate` verb, eval result schema (`EvalResult`), predictions schema,
  compatibility checks.
- **Tier 3 anchor:** `Metric` (`eval/base.py`).
- **Outbound deps:** `project`, `data` (concept-level + the public slice loader — see
  the seam note below), `mlflow.artifacts.eval`, `mlflow.artifacts.predictions`,
  `utils.io`, `utils.hashing`.
- **File renames (committed in §00 appendix):**
  `eval/snapshot.py` → `eval/eval_dataset.py`;
  `eval/loader.py + eval/loading.py` → `eval/_load.py`;
  `eval/publish.py` → `eval/prepare.py`;
  plus new `eval/results.py` (`EvalResult` / `EvalIndex` / `Predictions`) and `eval/checks.py`.

---

## Q1 — Does the `split_view` EvalDataset kind survive, given data's split-at-load?

**Decision: (B) — keep both `EvalDataset` kinds (`external` + `split_view`), but
`split_view` delegates realization + slice-hashing to data's public slice API.**

### The root issue (recorded for future reference — the "north star")

`EvalDataset`'s `kind ∈ {split_view, external}` enum secretly bundles **three
orthogonal axes** into two values, then patches the gaps:

| Axis | Question | `split_view` | `external` |
|---|---|---|---|
| **Provenance** | Where do the rows come from? | derived from a training `Dataset` | imported independently |
| **Ownership** | Owns bytes, or a view? | view (realized on demand) | owns parquet bytes |
| **Role** | Train or eval substrate? | eval | eval |

Encoding orthogonal axes as one enum is why the current design feels "meshed":
the symptoms — `split_view`'s dual identity (recipe-hash *and* realized-content-hash
plus a wall of consistency validators), the triplicated `_realize_split_view_frame`,
the `_json_hash` private-import leak (already half-fixed by §13.8), the
`of_data_snapshot_id` string field — are all shadows of **one missing abstraction**:
a named, shared "immutable, content-addressed, hash-keyed table" + a named
"how it came to be" (lineage).

**The clean (north-star) model**, documented but NOT built now:

```
# substrate — one type, content-addressed, hash-keyed
Table(table_id, hash_key, target_column?, schema_hash, content_hash, lineage)

# lineage — composable edges, NOT a kind enum
Lineage = MaterializedFromSource(...)   # today's training Dataset
        | SliceOf(parent, split_id_col, ranges)   # today's split_view
        | ExternalImport(provenance)              # today's external
        | AugmentationOf(parent, added_columns)   # today's augmentation

# role (train vs eval) is a CONSUMPTION concern, decided by the reader,
# not baked into a separate type family.
```

Under it, "training Dataset" and "EvalDataset" stop being types and become roles;
`split_view` dissolves into "consume `SliceOf(training_table, test_ranges)` in the
eval role." `of_data_snapshot_id` is just `SliceOf.parent` promoted from an
implicit string field to a first-class lineage edge.

### Why we take (B), not the full rebuild

- **Effort fits the forward changes.** B is contained to `eval/_load.py` +
  `eval/prepare.py`: swap eval's hand-rolled bucket realization for a call to
  data's `load_dataset_by_id(..., split_range=)`, whose integrity comes from the
  content-addressed `of_dataset_id` + data's own load-time manifest validation
  (see Q2). The recipe-based `eval_dataset_id` and the 909-line `evaluate()`
  caching/TOC model are **untouched**.
- **Removes the only trigger that has actually landed** — duplicated slice
  realization (open-questions trigger (d)).
- **Strictly better than (A) "no change":** A keeps the duplication while still
  owing the rename + gate work, so it saves almost nothing.
- **Pre-pays the hard part of the rebuild.** Once `split_view` delegates "what is
  a slice" to data, eval no longer re-implements it; if a third byte-owning
  artifact family ever appears (trigger (a)), the jump to the lineage model is a
  promotion, not an untangling.
- **(C) collapse-to-external-only and the full rebuild are PAUSED** — they rewrite
  the working caching key + runner contract and re-introduce the unifying noun the
  Six-Nouns vocabulary retired, with no current functional pain forcing it.

**§13.8 resolution:** the 🟡 resolves to a *concrete change* (lineage-delegation),
not a no-op. The full substrate+lineage+role unification is the named follow-on,
re-opened when a byte-owning third artifact family appears (trigger (a)) — see the
seam-thickness tripwire below.

### Dependency seam: `eval → data` (one-way, watched)

Import-graph reality (verified 2026-05-24):
- `eval → data` is the **only** direction. Already 10 imports across 6 files;
  B adds `load_dataset_by_id` along this established, §8.4-sanctioned edge.
- The single `data → eval` back-edge today is `data/pipeline.py`'s deferred
  `from automl.eval.loader import load_evaluation_spec` inside
  `_validate_evaluation_columns` — the pre-flight gate (carry-back #2). Moving the
  gate eval-side (resolved later in this sub-spec) **deletes that edge**, leaving
  the graph strictly acyclic. **No circular-import risk; B improves the graph.**

**Tripwire (record for unification re-evaluation):** B gives eval a *runtime
data-loading* dependency on data (`load_dataset_by_id`), thicker than the prior
types-and-hashing-only seam. If eval grows to depend on progressively more of
data's loading internals, that thickening is the concrete, watchable signal that
the substrate+lineage+role unification has begun paying rent — the operational
form of open-questions triggers (a)/(d). Keep the `eval → data` import list in
§8.4 enumerated so the seam stays a known surface, not a tangle.

**Carry-back to §00 §8.4:** add `load_dataset_by_id` (data's public slice loader)
to the enumerated "what `eval` imports from `data`" list.

---

## Q2 — Under delegation, what does the `split_view` identity record?

**Decision: (B) — recipe-only identity for `split_view`.**

Grounding: `compute_eval_snapshot_identity` already derives the `split_view` id
**purely from the recipe** (`kind, target_column, hash_key, of_dataset_id,
split_id_col, buckets`), with `schema=""`/`content=""` for the identity. The
realized-frame `schema_hash`/`content_hash` are computed separately at
manifest-build time and re-verified by re-realizing + re-hashing on every load —
the "dual identity" smell.

**What guarantees integrity under recipe-only.** NOT data's `verify_loaded_slice`
(that is the **L3** validator — it checks a *trial's* loaded slice against its
`TrialDataContract`, and an eval split_view dataset has no such contract). The real
guarantee is simpler and stronger: `of_dataset_id` is **content-addressed** (data's
`Dataset` id is a hash of its content — `ComponentHashes`), so *same id ⟹ same
bytes*. When eval calls `load_dataset_by_id(of_dataset_id, split_range=buckets)`,
data validates the loaded `Dataset` against its own manifest (**L2 loaded↔manifest**),
and the bucket filter is deterministic. So the realized eval slice is fully pinned
by `(content-addressed id, buckets)` with no eval-side hash needed. The "parent
republished under the same id with different content" failure mode is *impossible* —
different content would produce a different id. So:

- **`split_view`**: `eval_dataset_id` stays recipe-derived (unchanged). **Drop the
  realized-frame `schema_hash`/`content_hash` from the manifest**; delete the
  split_view content-rehash + consistency-validator wall in `eval_dataset.py`.
  Integrity comes from data's content-addressed id + L2 load-time validation.
- **`external`**: untouched. It owns bytes, so `content_hash` **is** its identity.

**Prepare path delegates too (not just load).** `prepare_eval_split_view` today
reads the full parent parquet and re-hashes the realized frame at publish time —
the same realization Q1 removes from the load path. Under recipe-only it becomes
**recipe-only at publish as well**: validate cheap *metadata* (parent `dataset_id`
resolves; `split_id_col` present in the parent manifest; buckets well-formed —
`0 ≤ start < end ≤ 100`, no overlap, carried forward from `_normalize_buckets`),
then write the recipe manifest. **No full-frame realization or hashing at publish.**
`validate_eval_manifest_v1` is updated so split_view manifests no longer carry/require
`schema_hash`/`content_hash` (external still does).

**Intentional behavior shift (recorded, not hidden):** empty-bucket detection moves
from **publish-time → first-load/evaluate-time**. Today `prepare_eval_split_view`
raises if the realized frame is empty; under recipe-only that's discovered when the
slice is first loaded. Consistent with the split-at-load philosophy; accepted.

**Rationale.** Q1 removed duplicated *realization*; Q2 removes the duplicated
*hashing/identity* that was the actual dual-identity smell. Eval re-hashing is the
redundant guard `feedback_no_redundant_guards` says to drop, since data's
content-addressing + L2 already pin the bytes. The two kinds become conceptually
honest: `split_view` is identified by *what it selects* (recipe), `external` by
*what it contains* (content) — correct, since a view has no bytes of its own to
serve as identity.

**Behavioral note (intended):** two different bucket recipes that realize
byte-identical rows get *different* `eval_dataset_id`s. Correct — for a view the
recipe is the identity; coincidental row-equality must not collide.

**Cross-spec dependency on sub-spec 05 (recorded):** recipe-only integrity assumes
data's `load_dataset_by_id` runs **L2 (loaded↔manifest)** validation by default; if
05's load does not, eval calls the L2 validator explicitly after load. Also,
`split_range` must accept **multiple disjoint bucket pairs** (e.g. `((80,90),(95,100))`),
not just one contiguous range. Both are carried back to sub-spec 05 (see open-questions).

---

## Q3 — Scope of "snapshot" retirement in `eval/`

**Decision: (A) — full retirement, matching `data/` §05 Q1. Clean cut, no back-compat.**

Every "snapshot" symbol / path / tag in `eval/` becomes "dataset" / "eval dataset":

| Today | After |
|---|---|
| `EvalSnapshotIdentity` | `EvalDataset` (Tier 2 export per §00 §8.4) |
| `EvalSnapshotKind` | `EvalDatasetKind` |
| `eval_snapshot_id` (field/param/manifest key) | `eval_dataset_id` |
| `compute_eval_snapshot_identity` | `compute_eval_dataset_identity` |
| `prepare_eval_snapshot` / `LoadedEvalSnapshot` | `prepare_eval_dataset` / `LoadedEvalDataset` |
| `eval_snapshot_gcs_paths` / `snapshot_name` / `snapshot_hash8` | `eval_dataset_gcs_paths` / `dataset_name` / `dataset_hash8` |
| **`of_data_snapshot_id`** (carry-back #3) | **`of_dataset_id`** |
| GCS path `…/eval/snapshots/<name>` | `…/eval/datasets/<name>` |
| MLflow tag keys `eval_snapshot*` | `eval_dataset*` |

Rationale: consistent application of §05 Q1 + §00's vocabulary mandate ("Snapshot"
retired; the noun is Dataset). Partial retirement would leave eval as the lone
domain where the retired word lingers — including the split-brain of
`eval/datasets/…` sitting under a folder still named `snapshots/`. Clean-cut /
no-back-compat is locked (`feedback_no_back_compat`); unreadable old eval
manifests/tags/paths is an accepted cost. **Carry-back #3 closes here** as one line
item of A.

---

## Q4 — Eval-column pre-flight gate home (carry-back #2)

**Decision: two checks at two lifecycle points; one pure predicate; eval owns the
logic; callers invoke it. Closes carry-back #2.**

The two checks are NOT redundant — different timing, different failure modes:

| | When | Sees | Catches |
|---|---|---|---|
| **early** | dataset materialize (`automl data build`, preview, dry-run) | the freshly-built schema | config↔data mismatch **before any model.py is written** |
| **pre-fit** | runner, before fit | the loaded eval frame | EVAL gaining a required column **after** the dataset was materialized (config drift) |

Both check **realized columns**, not config — the only way to catch a column config
declared but quality-filters dropped.

### Three-layer validator model (confirms sub-spec 04; no change)

1. **`validate/` framework** owns the vocabulary (`Issue`, `ValidationReport`,
   `Severity`, `_safe`) + the grouped orchestrators (`project`/`model`/`proposal`).
   **No registry route** — checks are plain function calls (§04 Q1); the only
   registry-ish thing is the project-side `PROJECT_CHECKS` dict.
2. **Each domain's `checks.py`** *defines* its checks (`fn(...) -> Iterable[Issue]`).
3. **Callers** invoke at the right point — CLI verbs, and lifecycle orchestration
   (runner, the data-build verb). The eval-column gate is a *targeted* check called
   directly (like `validate.model`), NOT routed through `validate.project`.

### Concrete shape (one source of truth)

- **`eval/checks.py`** holds the pure predicate
  `missing_eval_columns(*, spec, columns, target) -> list[str]`. Two surfaces on top:
  - `EvalSpec.validate_columns(df, target)` — **raising** (existing method; runner
    pre-fit + used internally by `evaluate()`).
  - `eval/checks.py:check_eval_columns(*, spec, columns, target) -> Iterable[Issue]`
    — **non-raising**, report-style.
- **Caller 1 (early):** the **surface-layer data-build verb** calls
  `check_eval_columns` against the materialized schema and reports — **whether the
  dataset was freshly built or a cache hit** (it checks the resulting schema either
  way). Lives in the verb, **not the data domain** — so `data → eval` is never
  introduced (the verb imports both; the domain imports neither). Preserves the
  acyclic graph from Q1.
- **Caller 2 (pre-fit):** the runner calls `EvalSpec.validate_columns` before fit.

**Covers all three legacy call sites.** Today `_validate_evaluation_columns` fires at
three points in `data/pipeline.py`: fresh-build split (`:806`), existing-dataset
re-validation during materialize (`:1109`), and trial-time load (`:1200`). The first
two map onto **Caller 1** (the verb checks the materialized schema regardless of
build-vs-reuse); the third maps onto **Caller 2** (runner pre-fit). None is dropped.

The legacy "exactly 1 target column" sub-check is a registry invariant → moves to
data/validate, not eval's concern (eval only needs the target name).

**Deferred:** a unified `automl validate` target covering data↔eval consistency
against a materialized dataset — only if demand appears (`feedback_extension_points_follow_demand`).

---

## Q5 — Eval verb surface: two entry points, `session` convention

**Decision: (A) — keep both entry points; delete `eval/runner.py`; rename the
lightweight `run()` → `evaluate_frame()` in `eval/evaluate.py`; `session`
convention throughout.**

- **`evaluate(*, session=None, model_run_id, eval_dataset_id, …) -> EvalResult`**
  — the full stateful verb (loads eval dataset + model, predicts, caches
  predictions to GCS, writes report + TOC, logs scalars). `automl eval compute`.
  The 909-line caching body is carried forward intact (Q1/Q2 leave it untouched);
  only `ctx → session` and `eval_snapshot_id → eval_dataset_id` change. **The
  `_model` / `_model_feature_registry` injection params are preserved** — the runner
  (`runner/_execute.py`) passes the already-loaded model + registry to skip an MLflow
  artifact download. Returns `EvalResult` (Q6), not a separate `EvaluateResult`.
- **`evaluate_frame(*, y_pred, df, spec=None, target_col=None, session=None)`**
  — pure in-process metric computation (no MLflow / GCS / model load). Was
  `eval/runner.py:run()`. Relocated beside `evaluate()`; renamed to avoid the
  collision with the top-level `runner/` domain.
- **`eval/runner.py` is deleted** (not in §00's `eval/` layout).

Rationale: the two have genuinely different contracts — a pure computation vs a
stateful logging verb with caching. Collapsing them (B) would add a second mode to
the most complex file in the domain for no gain (cuts against simple/native).
Both adopt `session: Session | None = None` per sub-spec 01.

---

## Q6 — Type/schema inventory + naming consolidation (mirror `data/`)

**Decision: consolidate the whole eval type family to mirror data's §05
vocabulary. Net effect — four types removed, survivors line up 1:1 with
`Dataset` / `LoadedDataset` / `DatasetIndex`. No new concepts introduced.**

Trigger: `EvaluateResult` vs `EvalResults` is a confusing near-collision — and it's
not the only inconsistency. Full review surfaced three:
1. **Result clash:** `EvaluateResult` (runtime return) and `report.json` (persisted
   twin) are two shapes for one concept.
2. **Identity/Pointer split:** eval has two types per artifact (`…Identity` +
   `…Pointer`) where the pointer just bolts GCS URIs + `cached` onto the identity.
   **Data dropped this** (§00 §299 — paths are `Dataset.*_gcs_uri` properties; the
   standalone `snapshot_gcs_paths` function retired). Eval still has the old split.
3. **Manifest-as-dict:** data treats its GCS manifest as `Dataset.to_dict()`/
   `from_dict`; eval hand-builds a separate manifest dict.

### Consolidated naming (before → after)

| Concept | Today | After | Action |
|---|---|---|---|
| metric ABC | `Metric` | `Metric` | keep |
| eval recipe | `EvalSpec` | `EvalSpec` | keep |
| eval-dataset identity **+ path properties** | `EvalSnapshotIdentity` + `EvalSnapshotPointer` | **`EvalDataset`** | **merge** — paths become `data_gcs_uri`/`manifest_gcs_uri` properties (mirrors `Dataset`); `eval_snapshot_gcs_paths` retired |
| loaded eval dataset | `LoadedEvalSnapshot` | **`LoadedEvalDataset`** | rename (mirrors `LoadedDataset`) |
| kind literal | `EvalSnapshotKind` | **`EvalDatasetKind`** | rename |
| augmentation identity **+ paths** | `AugmentationIdentity` + `AugmentationPointer` | **`Augmentation`** | **merge** |
| eval GCS manifest | hand-built dict | `EvalDataset.to_dict()` / `from_dict` | **drop separate schema** |
| per-(model, label) outcome | `EvaluateResult` + `report.json` | **`EvalResult`** (singular) | **merge into one** |
| run-level index of outcomes | `eval/manifest.json` TOC dict | **`EvalIndex`** | type it (mirrors `DatasetIndex`) |
| predictions | dict | **`Predictions`** | type it |
| compat errors | `ColumnMissing` / `DtypeMismatch` | keep | keep |

### The result type, settled

- **One `EvalResult` (singular).** `evaluate()` returns the same `EvalResult` it
  persists — no twin. `schema_version: int = 1` + `from_dict` (§02), lives only here.
  Fields = the report shape: `label, eval_dataset_id, eval_dataset_kind,
  predictions_uri, predictions_manifest_uri, augmentations_used, primary, metrics,
  computed_at, schema_version`.
- **Metric logging is cross-trial-stable (sub-spec 11 #4 — confirm; already true).**
  The whole locked metric set in `metrics` is logged to MLflow under **namespaced
  `<label>.<metric>` keys** (e.g. `holdout.auc`; legacy `eval/evaluate.py:588`), which
  are stable across trials regardless of any single trial's *current* primary — this is
  exactly the key `mlflow.experiment.top_n_by_metric` ranks on (02). `primary` on
  `EvalResult` is the trial's own primary-metric *name* = **provenance/display only**;
  the additional bare-`<primary>` metric log (legacy `evaluate.py:596`) is a per-trial
  convenience (read by `TrialSummary.primary_metric_value`), **not** the cross-trial sort
  key. No change to eval here — the namespaced logging already holds; recorded so it
  isn't dropped.
- **`mlflow_url` is dropped from the type entirely** — it is *not* pure derivation
  (it needs a live `client.get_run()` to map `run_id → experiment_id`), so a property
  on a domain object would both make a surprise network call and violate the
  mlflow-seam invariant (domain objects never call mlflow). Instead the CLI derives
  it on demand via an **mlflow-seam helper** (`mlflow.…run_url(run_id)`), using the
  `model_run_id` it already passed in. No field, no seam violation.
- **`cached` is the single runtime-only field**, populated by `evaluate()` at its
  return sites (it knows whether it recomputed or reused). Mechanism, stated to avoid
  a silent trap: `to_dict` omits it; `from_dict` defaults it to `False`. It is a
  *return-channel* signal about *this* `evaluate()` call — **not meaningful on an
  `EvalResult` loaded from a persisted report**, and no caller branches on it
  programmatically (CLI display only). Kept flat (vs. returning `(EvalResult, cached)`)
  because the blast radius is one display-only field.

### `EvalIndex` and `Predictions` shapes

- **`EvalIndex`** (was the `eval/manifest.json` TOC; mirrors `DatasetIndex`):
  `primary_label: str | None` + `evaluations: tuple[EvalIndexEntry, ...]`, where each
  entry carries `label, eval_dataset_id, kind, report_path, eval_dataset_manifest_uri,
  predictions_uri, predictions_manifest_uri, augmentations_used, computed_at`
  (today's `_TOC_ENTRY_KEYS`). `schema_version: int = 1` + `from_dict`.
- **`Predictions`** (Tier 2 export): the typed predictions artifact = `y_pred` +
  `hash_key` columns (parquet) with a sidecar manifest (`schema_version: int = 1` +
  `from_dict`) carrying `trial_run_id, eval_dataset_id, eval_dataset_kind, label,
  hash_key, augmentations_used`. Written/read by `mlflow/artifacts/predictions.py`.

### Path-property provenance (the Identity+Pointer merge)

Merging `…Identity` + `…Pointer` means `EvalDataset` / `Augmentation` must carry
enough **route context** to compute their GCS URIs as properties (mirroring how
`Dataset` does it per §00 §299). `EvalDataset` gains the route fields it needs
(`bucket`, `gcs_prefix`, `project_name`, `experiment_id`, `dry_run`,
`namespace`) and exposes `manifest_gcs_uri` always + `data_gcs_uri` (None for
`split_view`, which owns no bytes). **These are derivation context captured on the
type** (snapshotted from the bound state at publish time so URIs compute as
properties), **not parameters threaded through eval functions** — mode still lives
once on `session.dry_run` and `dry_run` here is just the bool that selects the
`dry_run/` path segment (no `run_mode`/`"full_run"` string; 10 §7.2 collapse).
`namespace` mirrors 02's bound field (the `--namespace` isolation prefix →
`Session.namespace`, sub-spec 01; renamed from legacy `route_namespace`) — a
full-universe segment orthogonal to the mode. `Augmentation` likewise exposes its
`manifest_gcs_uri` / `data_gcs_uri`. The old pointers' `cached` bool is **not** an
identity property — `prepare_*` returns `(EvalDataset, cached)` /
`(Augmentation, cached)` where the publish path needs to signal cache-hit.

### Homes (per §00 `eval/` layout)

- `EvalDataset`, `EvalDatasetKind`, `Augmentation`, `LoadedEvalDataset` →
  `eval/eval_dataset.py` (+ `_load.py` for the loaders).
- `EvalResult`, `EvalIndex`, `Predictions` → `eval/results.py`.
- Typed writers/readers → `mlflow/artifacts/eval.py` + `mlflow/artifacts/predictions.py` (§02).

### Notes

- **No `ComponentHashes` analog.** Post-Q2, eval identity is simpler than data's:
  `external` = schema+content+recipe hash; `split_view` = recipe hash only. A
  hash-wrapper would be overkill — keep the hash fields inline on `EvalDataset`.
- `schema_version` resets to `1` for all eval artifacts — clean cut, no back-compat.
- All renames are clean-cut (`feedback_no_back_compat`); old eval manifests/tags
  unreadable, accepted.

---

## Remaining surface (carried forward; constrained by parent specs)

No open decisions — recorded for completeness and the migration map.

- **`Metric` ABC + builtins (`Auc`, `LogLoss`, `ThresholdSweep`)** — carry forward
  unchanged (`eval/base.py` for `Metric`/`EvalSpec`, builtins in `eval/metrics.py`).
  `compute(df_test, y_pred, target_col)`, `__neg__` signing, `with_alias`,
  `required_columns`, `required_augmentations` all preserved. The §00 cross-cutting
  "do ABC arg shapes align across domains (`Metric.compute` vs `DataSource.load`)"
  question stays a **closeout** item, not 07's.
- **`EvalSpec`** — carry forward (`primary` + `metrics`, duplicate-name guard,
  `required_columns`/`required_augmentations` aggregation, the augmentation-join +
  `validate_columns` logic). `validate_columns` is the raising surface used by the
  runner pre-fit gate (Q4).
- **Augmentations** — carry forward as the renamed `Augmentation` (Q6). Untouched by
  Q1's delegation (they *add columns* by hash_key join, they don't *select rows*, so
  they don't overlap data's slice machinery). North-star: this is the
  `AugmentationOf` lineage edge.
- **`eval/checks.py`** holds: `check_model_eval_compatibility` +
  `ColumnMissing`/`DtypeMismatch` (was `eval/compatibility.py`); the project-EVAL
  module-shape checks `check_evaluation_module_exports` + `_probe_evaluation_shape`
  (per §04 line 86); and the Q4 eval-column predicate
  `missing_eval_columns` / `check_eval_columns`.
- **`eval/_load.py`** merges `loader.py` (`load_evaluation_spec`) + `loading.py`
  (`load_eval_dataset`). The `split_view` branch **delegates realization to
  `data.load_dataset_by_id(of_dataset_id, split_range=buckets)`** (Q1); integrity is
  data's content-addressed id + L2 load-time manifest validation (Q2) — no eval-side
  bucket realization, no eval-side re-hash.
- **CLI verbs** (§00 §11.1): `automl eval list` → `list_eval_datasets(session)`
  (lists published `EvalDataset`s for the route); `automl eval compute
  --model-run-id … --eval-dataset …` → `evaluate()`. Flag `--eval-snapshot` →
  `--eval-dataset` (Q3). `evaluate_frame` is library-only (no CLI verb).
- **`scalar_metric_records` / `is_scalar_value`** stay in `eval/base.py` — both are
  imported cross-domain (`validate/builtin/contract_checks.py` imports
  `is_scalar_value`; `evaluate.py` + tests import `scalar_metric_records`). They are
  not renamed; just confirmed as part of `eval/base.py`'s public surface.
- **`schema_version` guards inside the carried-forward `evaluate()` body flip to 1.**
  Today the report/TOC use `2` and `_read_eval_report` raises on `!= 2`; under the
  clean-cut reset (Q6) those guards become `== 1`. Old `2` artifacts are unreadable,
  accepted (`feedback_no_back_compat`).

### Migration file map (`automl_legacy/automl/eval/` → `eval/`)

| Legacy | New |
|---|---|
| `eval/base.py` | `eval/base.py` (`Metric`, `EvalSpec`) |
| `eval/metrics.py` | `eval/metrics.py` (builtins) |
| `eval/snapshot.py` | `eval/eval_dataset.py` (`EvalDataset`, `Augmentation`, identity) |
| `eval/loader.py` + `eval/loading.py` | `eval/_load.py` |
| `eval/publish.py` | `eval/prepare.py` (`prepare_eval_dataset`/`_augmentation`/`_split_view`) |
| `eval/evaluate.py` | `eval/evaluate.py` (`evaluate`, `evaluate_frame`) + `eval/results.py` (`EvalResult`, `EvalIndex`, `Predictions`) |
| `eval/compatibility.py` | `eval/checks.py` (+ module-shape + eval-column checks) |
| `eval/runner.py` | **deleted** (`run()` → `evaluate_frame()` in `evaluate.py`) |

Note: `check_evaluation_module_exports` + `_probe_evaluation_shape` currently live in
`validate/builtin/contract_checks.py`; they move **into** `eval/checks.py` (per §04),
and `validate.project` imports them back from there.

### Callers to update atomically (clean cut, no bridge code)

The renames in Q3/Q5/Q6 ripple to these out-of-domain callers; each must change in
the same cut (`feedback_no_back_compat` — no aliases, no dual-read):

| Caller | What changes |
|---|---|
| `cli/eval.py` | `--eval-snapshot-id` flag → `--eval-dataset-id`; `result.eval_snapshot_id` → `eval_dataset_id`; `result.mlflow_url` → derive via mlflow-seam helper |
| `tests/contracts/test_eval_snapshot_layout.py` | asserts `--eval-snapshot-id` + `EvaluateResult` export — update to new flag + `EvalResult` |
| `tests/unit/test_eval_public_surface.py` | asserts old `__all__` names (`prepare_eval_snapshot`, `load_eval_snapshot`, …) — update to `prepare_eval_dataset` / `load_eval_dataset` |
| `tests/unit/test_eval_snapshot.py`, `test_eval_publish.py` | import from `automl.eval.snapshot` + old symbol names — repoint to `eval/eval_dataset.py` + new names |
| `mlflow/artifacts/eval.py` | `_MANIFEST_ENTRY_KEYS` / writer key `eval_snapshot_id` → `eval_dataset_id` |
| `loop_context/proposer_packet.py` (→ `agent/proposer_context.py`, sub-spec 11) | reads `report["eval_snapshot_id"]` → `eval_dataset_id` |
| `core/project_context.py` | `from automl.eval.loader import load_evaluation_spec` → `eval/_load.py` (see project→eval note) |
| `runner/_execute.py` | passes `_model` / `_model_feature_registry` to `evaluate()` (preserved); consumes `EvalResult` |
| `eval/prepare.py` (was `publish.py`) | GCS-helper imports `from automl.data.pipeline import …` → `utils.io.gcs` directly (per §00 §8.4 outbound deps), not via data's re-export shims |

### Cross-spec carry-forwards (recorded in open-questions)

- **→ sub-spec 05 (Data):** `load_dataset_by_id` must (a) run **L2 (loaded↔manifest)**
  validation by default (Q2's recipe-only integrity leans on it; else eval calls L2
  explicitly), and (b) accept **multiple disjoint bucket pairs** in `split_range`.
- **→ sub-spec 01 (Project):** `core/project_context.py` exposes an `evaluation_spec`
  property that imports `load_evaluation_spec` — a **project→eval** edge. Whether the
  project context should re-expose eval-spec loading is sub-spec 01's call; eval keeps
  owning `load_evaluation_spec` regardless.
