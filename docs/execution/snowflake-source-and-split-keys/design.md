# Snowflake source & split keys — design

**Status: IMPLEMENTED 2026-06-04.** Approved by wendao + Claude after four
review rounds + final alignment pass, then landed in four code steps with
post-landing fixes. This remains the source of truth for the effort's decisions;
[`README.md`](README.md) is the decision-free front door and
[`plans/README.md`](plans/README.md) is the execution ledger. Code is the source
of truth for current behavior — file references describe the tree as of this
date.

The design came out of a long working discussion; every decision carries its
**why** so it doesn't get relitigated in passing.

1. [The goal, grounded in principles](#1-the-goal-grounded-in-principles)
2. [The state model: three layers](#2-the-state-model-three-layers)
3. [Two identities: recipe and content](#3-two-identities-recipe-and-content)
4. [The flow: one materialize() call, end to end](#4-the-flow-one-materialize-call-end-to-end)
5. [The drift matrix](#5-the-drift-matrix)
6. [Where every record lives](#6-where-every-record-lives)
7. [Keys: unique_key and split_group_key](#7-keys-unique_key-and-split_group_key)
8. [SPLIT_PCT: the split column](#8-split_pct-the-split-column)
9. [SnowflakeSource: the contract](#9-snowflakesource-the-contract)
10. [Connector: snowflake-connector-python](#10-connector-snowflake-connector-python)
11. [Validation summary](#11-validation-summary)
12. [Flexible splits: serializable predicates (step 4)](#12-flexible-splits-serializable-predicates-step-4)
13. [What this changes vs. today](#13-what-this-changes-vs-today)
14. [Implementation plan](#14-implementation-plan)
15. [Open items](#15-open-items)

---

## 1. The goal, grounded in principles

What the data layer must guarantee, and which repo principle each comes from:

1. **Trials are comparable** — every trial in an experiment runs against one
   pinned, immutable dataset (*experiment pins its data snapshot*). Ten loop
   iterations that could each see different data are ten results that can't
   be compared.
2. **Everything reconstructs from the two stores** — MLflow holds the
   record/metadata, GCS holds the heavy bytes (*two stores, three levels*).
   No third source of truth, nothing findable only by magic paths.
3. **Warehouse work is explicit** — a Snowflake pull runs 5–10 minutes;
   nothing queries or rebuilds the warehouse unless the user asks.
   **No hidden state, no automatic "magic" logic**: the harness makes drift
   *visible*; only the user makes it *resolve*.

Everything below is these three, mechanized.

## 2. The state model: three layers

```
LAYER 1: Snowflake base table          ← the RAW snapshot (warehouse)
   built by:  generated DDL (user's base_table.sql SELECT + injected SPLIT_PCT)
   changes:   ONLY when missing (bootstrap) or --refresh-source
   cost:      expensive — the 5–10 min query over production tables

LAYER 2: Materialized dataset           ← the DERIVED training frame
   record →  MLflow (dataset.json: identity, hashes, recipe)
   bytes  →  GCS (data.parquet, feature_registry.csv)
   built by: pull from layer 1 (training_data.sql) × the recipe's transforms
   changes:  ONLY on --refresh-data (or first materialize)
   cost:     one SELECT against the existing base table — no rebuild

LAYER 3: Loads / trials                 ← read-only views of layer 2
   built by: reading GCS parquet, filtering SPLIT_PCT ranges
   changes:  never touch Snowflake, never write
```

The relationship that *is* the design: **layer 2 is a pure function of
(layer 1 × recipe)**. Everything else is detecting when one of the two inputs
changed — and telling the user rather than acting.

For file sources the same picture holds with layer 1 = the CSV/parquet file;
the file is its own "base table" and the layer-1 verbs are no-ops.

## 3. Two identities: recipe and content

Two hashes, answering two different questions at two different costs:

- **Recipe** — *"SHOULD the dataset be different?"* Computed from config
  alone, **without touching any source**. Definition is a mechanical rule,
  not a curated list: **the transitive set of inputs `materialize()` reads**
  — everything on the source (`base_table`, both SQL files **as content
  hashes, not paths** — editing a file is detected, renaming it isn't a
  change, `unique_key`, `split_group_key`, db/schema substitutions), the
  pipeline-relevant `DataSpec` fields (`exclude_cols`, `metadata_cols`, drop
  thresholds), and **`TASK.target`** (the pipeline protects and registers it,
  so it's an input even though it lives outside `DATA`).
  **Explicitly out:** `EVAL`, `RUN_CONFIG`, and notably `Splits` — splits are
  load-time views resolved into each *trial's* contract, never materialize
  inputs. GCS bucket/prefix are *where* the dataset lives, not what it is.
  All fields canonicalized before hashing (sorted sets, normalized paths) so
  cosmetic reordering isn't phantom drift.
- **Content** — *"IS the data different?"* The existing `identity_hash` over
  component hashes (frame content, schema, registry), knowable only after a
  pull. **This remains the dedup key and the only identity**: re-pulling
  unchanged data attaches to the existing version, never duplicates bytes.

The recipe is recorded **on the dataset record** (readable dict, not just a
hash) so drift reports name the fields that changed — "recipe drift:
`exclude_cols` changed" — instead of an opaque hash inequality. Recipe and
content are deliberately **many-to-one**: when an explicit refresh re-derives
identical content, the existing version's recorded recipe is updated
last-wins (the user explicitly refreshed; "this recipe currently produces
this content" is honest provenance, and the next default call matches
cleanly). The recipe is **not** part of identity — folding it in would mint
duplicate copies of identical bytes.

## 4. The flow: one materialize() call, end to end

```
materialize()
│
├─ 1. Active dataset exists?
│       ├─ recipe matches its record         → attach silently. DONE.
│       │                                       (no Snowflake, no GCS write — the default fast path)
│       └─ recipe drifts                     → LOUD WARNING with field diff, then attach anyway:
│                                               "recipe drift: exclude_cols changed since v3 —
│                                                running against v3 as pinned; --refresh-data to re-derive."
│                                               Repeats every call while drifted. Never blocks, never acts.
│
├─ 2. --refresh-data (or no dataset yet) → re-derive layer 2, so source.load() runs:
│       a. base table missing?               → run generated DDL          (bootstrap — nothing else possible)
│       b. --refresh-source?                 → run generated DDL          (rebuild — implies --refresh-data)
│       c. content sanity check, in SQL:     SPLIT_PCT == MOD(ABS(HASH(split_group_key)), 100) for all rows?
│             mismatch → ERROR naming --refresh-source                   (never auto-spend warehouse minutes)
│       d. fetch training_data.sql           → raw rows
│             dry-run: deterministic bucket sample — COUNT(*) on the base table,
│             k ≈ 100·dry_run_rows/total, pull WHERE SPLIT_PCT < k (same sample every run;
│             dry_run_rows is an approximate target here, exact nrows for file sources)
│
├─ 3. Pipeline transforms rows → frame; validate keys + SPLIT_PCT; compute content identity
│
├─ 4. Content matches an existing version?   → attach; update its recorded recipe (last-wins)
└─ 5. Else                                   → persist bytes to GCS, dataset.json to MLflow, mint v<N+1>
```

Design notes on the flow:

- **Warning, not error, on drift (settled on review):** the user may have
  deliberately staged a config change they know doesn't matter yet, or isn't
  ready to mint a snapshot. Blocking would be the harness deciding; the
  warning keeps the user the only actor while making drift impossible to
  miss.
- **No `prepare()` hook.** An earlier draft had a pre-load lifecycle stage;
  nothing needs to happen between it and loading, so steps a–c live *inside*
  `SnowflakeSource.load()`. The source keeps exactly its existing verbs
  (`load`, `identity`, `artifact_files`); "ensure my upstream exists" is the
  source's own business inside the one method that was always its job. File
  sources: `load()` reads the file, a–c don't exist.
- **Step 2c is a content check, not a provenance record.** An earlier draft
  stamped a `COMMENT` on the base table recording which key built it;
  rejected as a declared record we'd then have to trust. The invariant is
  checked **empirically against the actual table** — one cheap existence
  query per real pull — which catches strictly more (out-of-band rebuilds,
  hand-built tables) with nothing new to store. Any table satisfying the
  invariant is valid, whoever built it.
- **Why re-pull at all when config changes, given the config is logged:**
  we log the recipe and the *derived* result — not the raw rows. The
  persisted parquet is post-transform; the raw store **is** layer 1. The log
  detects staleness; only the raw store can resolve it. And "re-pull" is one
  `SELECT` against the already-built base table — never a rebuild, never a
  touch of production source tables.
- **Identity is order-insensitive** (change to `dataframe_content_hash`:
  sort the per-row hash list before fingerprinting — a canonical multiset;
  duplicates still count; no data is reordered). Files return rows in stable
  order for free; a Snowflake `SELECT` has no order guarantee, and without
  this the same unchanged table would mint a new version per pull. To be
  precise about the mechanism: identity does **not** sort by `unique_key` or
  reorder any data — each row hashes to a number, and the *list of numbers*
  is sorted before fingerprinting. The unique key plays no role in identity.
- **Snowflake dry-run is deterministic by buckets** (step 2d): same sample
  every run, so dry-run identity is stable too. File sources keep exact
  `nrows`; Snowflake trades exactness (`dry_run_rows` is approximated by
  whole buckets) for determinism — the right trade for a source where row
  order isn't guaranteed.

## 5. The drift matrix

| What changed | Detected by | What happens |
|---|---|---|
| nothing | recipe match | attach silently; zero I/O |
| `exclude_cols`, `metadata_cols`, thresholds, `unique_key`, `TASK.target`, `training_data.sql` | recipe vs. record | **warning + diff**, attach as pinned; user runs `--refresh-data` when ready |
| `split_group_key` | recipe vs. record; on refresh also step 2c fails | warning; on refresh, error names `--refresh-source`. *Why a rebuild:* the new key column almost certainly exists in the table — what's stale is the stored `SPLIT_PCT` **values**, frozen from the old key at DDL time; recomputing them is the rebuild. (A cheaper recompute-one-column path from the existing table is possible; deliberately not built — one DDL path until rebuild cost hurts in practice.) |
| `base_table.sql` text | recipe vs. record (content hash) | warning **recommending `--refresh-source`** (the table no longer reflects its definition; only a rebuild resolves it — the harness can't verify table-vs-SQL beyond the split invariant) |
| upstream warehouse data itself | **nothing config-side can detect this** — that's the point | user passes `--refresh-data`; content identity decides whether a new version actually mints |
| base table missing | step 2a | bootstrap automatically (nothing to destroy) |
| `EVAL` / `RUN_CONFIG` / `Splits` | — | never touches the dataset; splits re-cut affects future trials only |

## 6. Where every record lives

The two-store principle, applied without exception — **MLflow is the record,
GCS is bytes only**. (Today's code keeps `manifest.json` and
`dataset_index.json` in GCS; that is a deviation from the repo's own stated
principle and it makes the data layer's most important metadata invisible in
the MLflow UI. This design relocates it.)

```
MLflow, experiment overview run artifacts:        ← ALL metadata, browsable/searchable in UI
  datasets/<id>/dataset.json        ← the persisted Dataset: identity, hashes, recipe, gs:// URIs of its bytes
  datasets/<id>/source_trace/       ← executed SQL (base_table.executed.sql, training_data.executed.sql,
                                       post-substitution) + source_identity — already lives here today
  datasets/<id>/profile/            ← already lives here today

MLflow, experiment state:
  active-dataset pointer            ← already exists (set_active_dataset); id only — a true pointer.
                                      This IS the "what was used last time" record: a run that doesn't
                                      specify a dataset resolves pointer → its dataset.json

GCS (heavy bytes only):
  .../datasets/<id>/data.parquet
  .../datasets/<id>/feature_registry.csv
```

**Deleted:** the `datasets/index` and `datasets/latest` JSON mirrors, and the
GCS-side `dataset_index.json` + `manifest.json`. The folder structure *is*
the index — version ids are folder names, each folder owns its record; a
separate index is a second copy of truth that will eventually disagree with
the folders. "Latest" was duplicating record payload to answer a question
the active pointer answers with one id.

**Naming: no "manifest".** The file is the serialized `Dataset` object
(`dataset.to_dict()`), so it's named for the noun it serializes:
`dataset.json`. ("Data contract" was considered and rejected — that noun is
taken by the *trial-level* `RunDataContract`.) The rule generalizes across
the sweep: a persisted record is named for its noun — the eval layer's
manifests become `eval_dataset.json` and `augmentation.json` (serialized
`EvalDataset` / `Augmentation`).

The **trial-level `RunDataContract` is unchanged**: a pointer (dataset id +
hashes + slice ranges) that inherits the recipe by reference through the
dataset record. Nothing new rides on trials.

**Accepted trade (chosen, not slipped in):** hot paths (attach check, dataset
loads) move from raw GCS reads to the MLflow artifact API — proxied in
production, the path that previously had 500-on-missing sharp edges (now
mitigated by the list-first download helper). Small JSON reads; judged worth
it for principle-compliance and UI visibility, but it does put the data
layer's hot path on MLflow availability.

## 7. Keys: `unique_key` and `split_group_key`

Two declared keys, named for their jobs, replacing `hash_key` everywhere
(the old field conflated them — a latent bug, since a non-unique grouping
key would pass materialize and explode in eval's one-to-one join):

- **`unique_key` — required on every source.** Composite supported (tuple).
  The stable row identifier: eval joins, augmentations, `row_ids`, any
  future row-addressed operation. **Hard-validated at materialize**: columns
  present *and no duplicate key tuples* — loud error. Moving the guarantee
  to the ingestion edge makes everything downstream safe by construction;
  first contact will likely be a data-quality conversation on the fraud
  dataset, and that's the check working.
- **`split_group_key` — optional, defaults to `unique_key`.** The key whose
  hash assigns split buckets; declared separately only when grouping matters
  (rows are transactions, identity is `transaction_id`, splits must group by
  `user_id` so one user never straddles train/test). Grouped splits thereby
  become expressible for file sources too.

Both recorded on the dataset record. Uniqueness of `unique_key` is *not*
required of `split_group_key` (grouping is the point); eval-side one-to-one
validation stays at the eval edge where it belongs.

**Row fallback is removed** (`ROW_FALLBACK_HASH_KEY`, the `hash_key=None`
branch, and the test pinning it): content/position-derived buckets reshuffle
on any edit (the instability splits exist to prevent), the marker column was
never wired into the eval consumer (`frame.loc[:, [marker]]` → `KeyError`),
and nothing uses it. `unique_key` being required is the replacement.

## 8. `SPLIT_PCT`: the split column

`SPLITID` → **`SPLIT_PCT`** (`split_id_col` → `split_pct_col`; split helpers
and `split_report` renamed to match). Semantics unchanged: deterministic
0–99 assignment, hash of `split_group_key` mod 100; `Splits(train=[(0, 80)])`
still reads "80% of the data" — which is why the name fits: the only thing
anyone does with the column is declare proportions.

Who computes it differs by source; everything downstream is shared:

```
CSV / Parquet:  fetch rows → pipeline assigns SPLIT_PCT from split_group_key (pandas hash)
Snowflake:      fetch rows → SPLIT_PCT arrived from the base table (frozen at DDL time)
both:           → validate SPLIT_PCT (present, integer, 0–99)
                → validate unique_key (present, no duplicates)
                → content identity → attach-or-mint → persist
```

One universal post-split validation; the record notes the provenance
(`split: python(split_group_key=…)` vs `split: sql`). This also fixes the
current bug where a SQL-provided split column gets lowercased by column
standardization into a `splitid` *feature column* (leakage footgun) while
the pipeline recomputes its own.

**Recorded caveat:** Snowflake `HASH()` and pandas assign *different* buckets
for the same key values. Irrelevant within a project; a project migrating
CSV → Snowflake reshuffles split membership and old/new datasets are not
split-comparable. Inherent to two engines; documented, not designed away.

## 9. SnowflakeSource: the contract

```python
SnowflakeSource(
    base_table="FRAUD_TRAINING_BASE",                 # name only; lands at {database}.{schema}.{base_table}
    base_table_sql="data/queries/base_table.sql",     # the SELECT defining the base table (renamed from base_data_sql)
    training_data_sql="data/queries/training_data.sql",  # the SELECT pulling training rows
    unique_key="TRANSACTION_ID",                      # or a tuple for composite
    split_group_key="USER_ID",                        # optional; defaults to unique_key
)
```

### `base_table_sql` is a SELECT; the harness owns the DDL

**Contract change from the old system** (where the user wrote the full
`CREATE OR REPLACE`): `base_table.sql` contains *the SELECT that defines the
base data*. The source generates the DDL around it, **injecting `SPLIT_PCT`
from `split_group_key`**:

```sql
CREATE OR REPLACE TABLE {database}.{schema}.{base_table} AS
SELECT t.*, MOD(ABS(HASH(t.USER_ID)), 100) AS SPLIT_PCT   -- from split_group_key
FROM (
    <user's base_table.sql — joins, CTEs, filters, feature SQL>
) t
```

Enforced mechanically: rendered body must be a single statement starting
`SELECT`/`WITH`; a body already emitting `SPLIT_PCT` is a loud collision
error.

**Why SQL-side, frozen into the base table** (vs. computing in Python like
file sources): the bucket assignment becomes part of the warehouse snapshot —
deterministic reproduction from the database (every pull, full or filtered,
sees identical buckets by construction); first-class visibility to ad-hoc
human SQL; and future bucket-filtered pulls (`WHERE SPLIT_PCT < 10` — a
deterministic 10% sample) push down to the warehouse for free, which also
makes deterministic dry-run sampling a trivial follow-up.

**Why injected rather than user-written** (the old contract): the user
declares `split_group_key` once in `config.py`; no boilerplate
`MOD(ABS(HASH(...)))` line to write or forget, and no possible drift between
the declared key and the SQL — there is only one declaration.

**Why the harness owns the CREATE:** table creation has exactly one author —
it's what makes injection, bootstrap-on-missing, and `--refresh-source`
all clean. Rebuild cost is acceptable because the flag is rare and explicit
("the user knows the upstream has issues"); single-SELECT is the contract,
and anything that doesn't fit gets a project-owned escape hatch if and when
it appears.

### `training_data_sql` pulls rows

Scaffolds to `SELECT * FROM {database}.{schema}.{base_table}` — `SPLIT_PCT`
flows through. Arbitrary pull queries are fine; dropping `SPLIT_PCT` is
caught by the universal validation with a "carry SPLIT_PCT through from the
base table" message. Substitutions for both files: `{database}`, `{schema}`
(env), `{base_table}` (recipe). No case-mangling of identifiers (the old
implementation lowercased env values; dropped as surprising). Dry-run wraps
with `LIMIT`.

### Trace

`artifact_files()` returns the executed (post-substitution) SQL texts, logged
under `datasets/<id>/source_trace/` (§6) — the provenance trail shows
literally what ran.

## 10. Connector: `snowflake-connector-python`

The IO seam is `automl/utils/io/snowflake.py` (sibling of `gcs.py`):
env-driven params (`SNOWFLAKE_ACCOUNT`/`USER`/`PASSWORD` required;
`WAREHOUSE`/`ROLE` defaulting to `DATA_SCIENCE_WH`/`DATA_SCIENCE_ROLE`;
`DATABASE`/`SCHEMA` consumed by substitution), connection context manager,
`fetch_df(sql)` (Arrow → pandas), `execute(sql)`, `check_connection()`
(`SELECT 1`). Credentials in `.env` only, loaded by the existing
project-config `load_dotenv`; the agent never handles values.

**`snowflake-snowpark-python` is dropped from `pyproject.toml`.** Both
libraries are official and Snowflake-maintained; Snowpark is a client-side
DataFrame API + in-warehouse Python (UDF/sproc) layer *built on top of* the
connector (it's in Snowpark's own dependency tree). Our contract is "the
project owns SQL files; the harness executes SQL text" — on that path
Snowpark is a passthrough around the connector it ships with (the old
implementation used it exactly that way: `session.sql(sql).collect()`).
Carrying it costs lockfile surface — notably it pins `cloudpickle`, this
repo's model-packaging contract — for zero used capability. The choice is
encapsulated in the one seam file; if recipes ever author warehouse-side
Python, swapping is a one-file change.

## 11. Validation summary

At the **project edge** (`automl validate project`):
`project.connections.snowflake` stops being a pending warning and becomes a
live probe, mirroring GCS/MLflow checks — missing `SNOWFLAKE_*` env vars →
error listing which; else `check_connection()`, driver errors verbatim; both
SQL files exist on disk (the `TBD_` placeholder check already covers
content). Emitted only for Snowflake-backed projects. The pending-warning
language is pinned by a unit test and echoed in the setup/validate SKILL.md
and reference docs — all update in the same change.

At the **ingestion edge** (materialize):

| Check | Outcome |
|---|---|
| `unique_key` columns present, no duplicate tuples | **error** |
| `SPLIT_PCT` present, integer, 0–99 (all sources) | **error** |
| base-table split invariant (step 2c, Snowflake, on pull) | **error**, names `--refresh-source` |
| recipe drift vs. active dataset's record | **warning + field diff**, attach as pinned |

`split_report`/profile additionally surface unique-key cardinality, so 1:1
is visible, not just trusted. Interior code stays free of defensive checks,
per the validation-at-the-edges principle.

## 12. Flexible splits: serializable predicates (step 4)

**In scope — implemented as step 4** (settled on review: the design is done;
it was deferred only by scoping, not by open questions). It absorbs the ask
in [`../../archive/2026-06-04-time-based-splitting.md`](../../archive/2026-06-04-time-based-splitting.md)
(treated as a requirement; that note's open questions are answered here).
Note it is *not* an isolated runner concern: it touches the `Splits`
declaration (project), slice loading (`automl/data/registry.py`), the trial
contract's serialized split ranges, and eval split-view identity — which is
exactly why it ships inside this effort, while everything is already being
touched.

**The frame:** a split is a **named, durable row-criterion over an immutable
dataset** — and a bucket range is just one kind of criterion. Datasets
materialize unsplit (already true); the snapshot is immutable parquet; so
*any* pure column predicate is reproducible forever. Time-based splits need
**no new materialize-time machinery**:

```python
Splits(
    train = Where("application_date") < "2026-03-01",
    test  = (Where("application_date") >= "2026-03-01") & (Where("SPLIT_PCT") < 50),
)
```

Agreed properties:

- **Criteria are data, not code.** No lambdas — trial contracts and eval
  split-view identities must serialize and hash what a split *means*. Record
  representation *(amended 2026-06-04 at implementation, wendao-approved; an
  earlier draft said pyarrow DNF tuples)*: a small **nested JSON AST** —
  leaves `{"op": "<", "column": ..., "value": ...}`, composites
  `{"op": "and"/"or"/"not", "items": [...]}` — which expresses arbitrary
  nesting directly, with no DNF expansion. A thin `Where` builder emits it,
  and it compiles to pyarrow dataset *expressions* (also accepted by
  `read_parquet(filters=…)`, so push-down loses nothing).
  The op set (`== != < <= > >= in not-in is-null not-null` + and/or/not) is
  the whole needed vocabulary. The same criteria could later compile to a
  SQL `WHERE` against the Snowflake base table — one representation, two
  push-down targets.
- **`SPLIT_PCT` is an ordinary column.** No `Bucket` sugar —
  `Where("SPLIT_PCT") < 80` covers it. **The range API is hard-cut in step
  4**: `Splits` values become predicates only, and the example/scaffold
  configs update from `[(0, 80)]` to `Where("SPLIT_PCT") < 80`
  (forward-only, pre-rollout — one split vocabulary, not two).
- **Record, don't police.** Overlapping splits are legitimate methodology
  (full-data views, progressive train sets, deliberate reuse); the harness
  records exactly what each named split meant for any trial and enforces
  nothing about disjointness. (`unique_key` makes overlap *measurable* when
  anyone wants to check.)
- **Column availability is the only requirement.** Metadata vs. feature is
  the user's modeling call; a criterion referencing a missing column fails
  loudly at load. No special validation machinery.
- **Rolling/backtesting windows** are a family of named splits — the naming
  scheme already accommodates them.
- **Push-down is layout-dependent for speed, never correctness.** Row-group
  pruning pays when the parquet is clustered by the filtered column; sorting
  the persisted frame by `SPLIT_PCT` is deliberately **deferred to ship with
  the push-down reader** so layout and its exploiter land as one unit.

## 13. What this changes vs. today

Honest deltas, in one place:

- `materialize()` gains the fast path (attach-as-pinned) — today it re-pulls
  and re-hashes on **every call**, including editing-a-CSV pickup. After:
  file edits surface as nothing until `--refresh-data` *(uniform across
  sources — deliberate: warehouse pulls are minutes, and loop iterations
  must compare against one pinned snapshot)*.
- `--refresh-source` (previously dead) becomes real — upstream rebuild
  (base table for Snowflake, no-op for files; implies `--refresh-data`) —
  and gains a sibling `--refresh-data` (re-pull/re-derive).
- `Splits` ranges → `Where(...)` predicates (step 4, hard cut); trial
  contracts and eval split-view identities serialize the predicate AST.
- Snowflake dry-run: deterministic bucket sample replaces `LIMIT`
  (`dry_run_rows` becomes an approximate target there; file sources keep
  exact `nrows`).
- Dataset metadata moves GCS → MLflow; index/latest mirrors deleted;
  "manifest" → noun-named records (`dataset.json`, `eval_dataset.json`,
  `augmentation.json`).
- `hash_key` → `unique_key` (+ new `split_group_key`); row fallback deleted;
  new hard checks at materialize.
- `SPLITID` → `SPLIT_PCT` across ~35 files (mechanical; contract tests move
  with the shapes they pin).
- `dataframe_content_hash` becomes order-insensitive (one-time identity
  change for existing datasets — no migration; see ground rule below).
- `SnowflakeSource` becomes real; Snowpark leaves the lockfile.

## 14. Implementation plan

Four steps, each leaving the suite green; contract tests update with the
shapes they pin, in the same step.

**Step 1 — keys & naming cleanup (no Snowflake).**
`hash_key` → `unique_key` across data/eval/scaffold/docs/tests; add
`split_group_key` (defaults to `unique_key`); delete row fallback;
`SPLITID` → `SPLIT_PCT` (+ helper renames, `split_report`/profile sweep);
materialize-edge validation (unique_key present + unique; SPLIT_PCT valid).
Behavior-preserving for well-formed projects; loud for latent duplicates.

**Step 2 — dataset record & lifecycle.**
Recipe (mechanical-rule fields, canonicalized, SQL-as-content-hash) recorded
on the dataset record; record relocates to MLflow as `dataset.json` per
version folder (eval records renamed in kind); `datasets/index`/`latest`
mirrors and GCS index/manifest deleted; attach-as-pinned default with
drift-warning + field diff; `--refresh-data`; last-wins recipe update on
attach-after-refresh; order-insensitive `dataframe_content_hash`; refresh
flags plumbed through `load()` (no new hooks), replacing `--refresh-source`
in CLI/agent options/contract tests.

**Step 3 — Snowflake.**
`utils/io/snowflake.py` seam (connector-python); real `SnowflakeSource`
(SELECT-only enforcement, DDL generation + `SPLIT_PCT` injection, collision
error, substitutions, split-invariant content check, bootstrap/rebuild
inside `load()`, deterministic bucket-sample dry-run, executed-SQL trace
artifacts); live validation check (+ pending-language doc/test updates);
scaffold templates (`base_table.sql`, `training_data.sql`, config template);
Snowpark dropped from `pyproject.toml`; reference docs updated.

**Step 4 — flexible splits (§12).**
`Where` builder + serializable predicate AST (nested JSON record form, pyarrow
expression compile form); `Splits` hard-cut from ranges to predicates;
slice loading via predicate filter; trial contract + eval split-view
identity payloads carry the serialized predicate; example/scaffold configs
updated; range API removed.

**Tests.**
- *Unit*: seam mocked — SQL rendering/substitution, DDL generation +
  injection, SELECT-only and collision errors, invariant-check gating,
  bootstrap/rebuild gating, `LIMIT` wrapping, identity stability under row
  shuffle, key validation errors, drift-warning behavior, validation issue
  shapes. The two stub-pinning tests in `test_sources_breadth.py` are
  replaced.
- *E2E*: gated on `AUTOML_E2E=1` **and** `SNOWFLAKE_*` set (skip otherwise,
  matching `tests/e2e/_gates.py`), exercising `materialize()` against a real
  table via a throwaway `dev_`-prefixed project. `fraud_anomaly_detection/`
  and `payment_routing/` untouched except as eventual consumers.

**Ground rule for all steps:** no step deletes or migrates existing MLflow
runs, GCS objects, or warehouse tables — old state is wiped **manually by
wendao** for a clean slate; pre-existing warehouse tables are never replaced
without the explicit flag.

## 15. Open items

Settle during implementation; none blocking:

- **Exact recipe field list** — derived mechanically in step 2 from what the
  pipeline reads; documented on the record.
- **File-source `SPLIT_PCT` collision**: Snowflake errors loudly when the
  user's SELECT already emits the column; file sources today silently drop
  and recompute a same-named column. Proposal: error for symmetry. Decide in
  step 1.
- **Record/helper naming details** (`read_dataset_record` vs. alternatives)
  — step 2.
- **`Where` builder surface details** (operator coverage, repr, raw-vs-
  normalized column names in predicates) — step 4.

Resolved during the 2026-06-04 review rounds (kept for history): uniform
default-attach incl. file sources; table-COMMENT provenance **rejected** in
favor of the empirical invariant check; `prepare()` hook **rejected** —
folded into `load()`; auto-re-derive-on-drift **rejected** in favor of
warning + explicit refresh (which also dissolved the `recipe_hashes`-set
construct an earlier draft needed); hard uniqueness confirmed; warning (not
error) on drift; dataset metadata relocates to MLflow; "manifest" renamed to
noun-named records; rebuild cost accepted as rare+explicit; forward-only
with manual wipe by wendao. Fourth round: flags named
`--refresh-data`/`--refresh-source`; flexible splits pulled into scope as
step 4 (range API hard-cut); Snowflake deterministic bucket-sample dry-run
pulled into scope; active-dataset pointer confirmed as the retained
"last used" record (only the duplicating mirrors die); split_group_key
rebuild rationale clarified (stale frozen SPLIT_PCT values, not a missing
column). Final alignment pass confirmed: step 4 stays in scope in order
(1→4, with land-1-to-3-first as the known fallback if fraud timeline
presses); the agent loop never passes refresh flags itself (humans may, via
`experiment run`); example_homecredit churn accepted with live notebook
verification as a tail-end activity; MLflow-on-the-hot-path accepted.
