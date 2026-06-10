# Dataset-read reliability — design

**Status:** design captured from a working session (2026-06-10), not yet
finalized into `plans/`. The next session should pressure-test this, fill gaps,
and decompose into steps. Decisions below are recommendations with rationale,
not yet ratified.

---

## 1. Where `main` is today (the starting point)

The read path on `main` (branch `core/dataset-read-reliability` is cut from it)
is naive but correct:

- **`automl/utils/io/gcs.py: read_parquet(uri)`** — `blob.download_as_bytes()`
  then `pd.read_parquet(BytesIO(...))`. A **single-shot, whole-object** download
  into RAM: one HTTP GET on one connection, **no** resumable/chunked transfer,
  **no** checksum enforcement, **no** retry/timeout config passed to the client.
  `read_csv` is the same via `read_bytes`.
- **`automl/mlflow/experiment/artifacts.py: read_dataset_frame / read_registry`**
  — thin wrappers that re-raise any read failure as `StorageError`.
- **`automl/data/registry.py: load_dataset_by_id`** — reads the full frame +
  registry, calls `validate_loaded_dataset` (verifies n_rows / n_columns /
  `data_content` / `feature_registry` / `schema` hashes against the dataset
  manifest), then **slices** to the requested split and returns a `LoadedSlice`
  (the full frame is dropped; only the slice is retained).
- **`automl/data/contract.py`** — `dataframe_content_hash`, `schema_hash`,
  `validate_loaded_dataset`, `verify_loaded_slice`. This is where the
  content-addressing lives. Already imported *downward* by the eval layer — fine
  per layering.
- **`automl/runner/trial.py: _trial_data_contract`** — loops every split in
  `RUN_CONFIG.splits.predicates`; for each split that isn't the already-loaded
  fit split it calls `load_dataset_by_id(..., split_name=name)` → **another full
  download+parse**, just to compute that slice's `n_rows` + `content_hash`.
- **`automl/runner/serving_validation.py`** — runs the post-fit validation in a
  **`subprocess.run(...)`**; inside that child process it calls
  `load_dataset_by_id(dataset_id, split_name=eval_split)` → **another full
  download+parse, in a separate process** (cannot share parent memory).

**Per-trial read count** (neobank, splits `train`/`train_known`/`test`/`oot`):

| # | Site | For | Process |
|---|------|-----|---------|
| 1 | `trial.py` fit load | `train` | parent |
| 2 | `eval/_load.py` (eval) | `test` | parent |
| 3 | `_try_log_train_eval` | train known-only diagnostic | parent |
| 4–6 | `_trial_data_contract` | re-read `train_known`, `test`, `oot` for slice hashes | parent |
| 7 | `serving_validation.py` | `test` again | **subprocess** |

≈6–7 full multi-GB downloads + parses per trial; each ~63–73s.

**Not on `main`:** the outer retry band-aid (`_read_and_verify_dataset`) exists
only on the `neobank_ncm_v3_replicate` branch and is **deliberately not ported**.
This design replaces the need for it.

**No existing disk cache to reuse.** What looks like caching today is *not* a
local-bytes cache: `eval/evaluate.py: _load_cached_result` reuses an
already-logged eval result *from MLflow* (idempotency), and
`mlflow/client.py` keeps an in-memory `run_id → experiment_id` memo. MLflow's own
`download_artifacts` lands files in its managed temp location, not ours. The
content-addressed disk cache here is **net-new** — don't go hunting for an LRU
helper to extend.

## 2. Diagnosis (why it fails)

Two independent dimensions multiply:

- **Redundancy** — the same immutable bytes are fetched ~7×. Each fetch is an
  independent chance to hit a transient, and each costs full download+parse time.
- **Fragile single fetch** — `download_as_bytes()` holds one connection open for
  the entire multi-GB transfer with no resumability or checksum. Observed
  transients (all rare per-read, common in aggregate, on and off VPN):
  - **corrupted bytes** — frame arrives the right shape but wrong content; caught
    one level up by the manifest `data_content` hash as `DataError`. A fresh read
    is deterministic and matches → transient, not drift.
  - **oauth2 `/token` timeout** / connection reset → surfaces as `StorageError`.
  - VPN (Check Point SASE TLS interception) widens the reset window — see memory
    `gcs-vpn-throughput`, `vpn-sase-tls-ca-bundle`.

Size matters only *indirectly*: a bigger blob = a longer-lived single connection
= a bigger interruption window, and we reopen that window ~7×.

## 3. The design — three layers that converge on one primitive

The key insight: a **content-addressed local cache** at the read seam collapses
most of the problem. The first read of a trial does a robust populate; every
subsequent read — in-process *and the subprocess* — becomes a local-disk read.

```
read_dataset_frame(uri)                              ← data/mlflow read seam
  └─ blob_cache.get_or_populate(key = dataset_id + content_hash)   ← utils/io (generic CAS file cache)
        ├─ HIT  → read local parquet (no network, no transient)
        └─ MISS → robust download (resumable + crc32c + tuned retry)
                  → atomic write (tmp → rename) → verify manifest hash
                  → LRU-evict if over size cap
contract builder: slice the one in-hand frame in memory       ← removes reads #4–6
serving-validation subprocess: cache HIT on local disk        ← removes the read that failed
automl data cache {list,prune,clear}                          ← thin CLI verb
```

### Layer 1 — robust populate (replaces option-3 "use the client natively")

On a cache miss, fetch with the storage client's native robustness instead of a
bare `download_as_bytes()`:

- **`blob.download_to_filename(tmp_path, retry=<tuned>, checksum="crc32c")`** —
  resumable/chunked transfer with HTTP-layer exponential backoff, and checksum
  validation against GCS's stored checksum (catches transport corruption at the
  source).
- **Tune the retry** beyond the conservative default: a predicate/deadline that
  also covers the oauth2 token-refresh timeout we actually see.
- **Defense in depth:** still verify the parsed frame against our **manifest
  content hash** once (catches "right bytes for the wrong object" / decode
  issues that a transport checksum can't).
- **Stream to a file**, not into a `bytes` in RAM — avoids holding raw bytes +
  parsed frame simultaneously.

This is "better than option 3" = native robustness **+** tuned retry **+** a
final manifest re-verify.

### Layer 2 — read-once in process (the one easy in-memory win)

The contract builder (reads #4–6) re-reads only to hash slices. Fix it to slice
the **one in-hand frame** in memory using each split's existing predicate mask
(`RUN_CONFIG.splits.predicates`), computing `n_rows` + `dataframe_content_hash`
on the masked frame. `_trial_data_contract` already receives `loaded_fit`; it
needs the full frame (or the cache) in hand.

- **Keep slice-hash semantics identical** — hash the *sliced* frame, not the
  full one, so existing `SliceContract` / tag-lineage verification is unchanged.
- **Do not** thread a full-frame object through the eval domain (crosses
  boundaries, changes signatures). With the cache in place, the eval/diagnostic
  re-reads are already cheap local reads; only the contract builder is worth an
  explicit in-memory slice. Bound retained memory by keeping slices + hashes and
  releasing the full frame.

### Layer 3 — the cache is the cross-process bridge

The serving-validation **subprocess** (read #7) is the one read that failed and
that in-memory reuse can *never* cover — it's a different process. The
content-addressed disk cache is exactly what lets it reuse the parent's already-
fetched bytes: same `dataset_id` → same cache entry → local read, no network.

## 4. Layering (where each piece lives)

Respecting the repo principle (generic core low; concern at one altitude;
no sideways duplication):

- **Generic mechanism → `automl/utils/io/` (leaf).** A content-addressed file
  cache: `key → file`, with atomic writes (tmp → rename), size-cap **LRU-on-
  write** eviction, touch-on-read for recency. Knows *nothing* about datasets —
  reusable for any heavy artifact later.
- **Dataset-read policy → the data read seam**
  (`registry` / `artifacts.read_dataset_frame`). Decides "key = dataset id +
  content hash; verify against the manifest hash already checked here." Sits
  **below** every domain call site (eval, runner, contract), so all callers
  benefit without knowing the cache exists.
- **Hashing stays in `automl/data/contract.py`.** `dataframe_content_hash` is a
  data-domain concept and eval already imports it downward — that's the layering
  working, not a violation. **Do not** move it to utils.
- **CLI `automl data cache {list,prune,clear}` → `automl/cli/`**, a thin wrapper
  over a library function (surfaces stay thin).

## 5. Retention policy

- **Content-addressed key** (`<cache_root>/datasets/<dataset_id>/data.parquet`,
  + registry). A run pins one dataset → usually one entry. **General pool, not
  per-experiment** (identical bytes never stored twice).
- **LRU-on-write, size-capped.** Each populate (the one event that grows the
  cache) checks total size and evicts least-recently-used entries until under
  cap. This is the only "proactive" hook available without a daemon.
- **No TTL / age-based eviction** — datasets are immutable and a run pins one;
  age tells you nothing and could evict the dataset you're about to reuse.
  Recency (LRU) won't.
- **Explicit `automl data cache {list,prune,clear}`** for manual / CI control.
- **Atomic populate** (download to temp → verify → rename) so a concurrent
  reader never sees a half-written file; a reader either sees a complete file or
  falls through to re-download.
- **Proposed defaults (confirm in plan):** cache root `~/.cache/brigit-automl/`
  (never in the repo / `projects/`), size cap ~20 GB (a couple of full datasets).

## 6. Net effect

≈7 fragile network reads/trial → **1 robust populate, the rest local-disk
reads**. Faster, and the transient surface mostly disappears rather than being
retried around. The `neobank_ncm` branch's `_read_and_verify_dataset` retry is
**retired** (no longer load-bearing).

## 7. Companion: Snowflake `skip_live_check` → `RUN_CONFIG`

Small, in scope because it's the same "run reliably off-VPN" theme and your
preferred shape.

- **Today (neobank branch only):** `SnowflakeSource.skip_live_check: bool` gates
  the live `SELECT 1` in `validate_project`; `checks.py` reads
  `source.skip_live_check` (with a `probe_snowflake=` override plumbed through).
  Operational state bolted onto the *source* — and `main` doesn't have it at all.
- **Target:** make it a `RUN_CONFIG` run-knob (operational, like
  `serving_validation_seconds`), e.g. `RUN_CONFIG.skip_snowflake_live_check` (or
  a small `validation`/`connectivity` sub-config). `validate_project` reads it
  from run config; the env-var and SQL-file checks still run; the live probe is
  the only thing gated; emit the same skip warning. Keep it **out of**
  `identity()` / `recipe_identity()` — it must never change the dataset hash.
- Note the `probe_snowflake` override is still useful for the CLI/tests to force
  the probe on/off regardless of config.
- **Follow-through:** once the flag moves, update the projects that set it today
  (`projects/neobank_ncm/config.py` uses `SnowflakeSource(skip_live_check=True)`)
  to the new `RUN_CONFIG` field, and remove the source flag. This is part of the
  `neobank_ncm_v3_replicate` rebase when this effort lands on `main`.

## 8. Open questions for the next session

- Exact cache root + size cap (defaults proposed in §5).
- Cache the **parsed frame** form or the **raw parquet bytes**? (Proposed: store
  the parquet file; parse on read — parse cost remains but is local. Revisit if
  parse dominates.)
- Whether `read_registry` (small CSV) is worth caching or only the heavy parquet.
- Concurrency: multiple trial processes sharing the cache dir — is atomic
  rename + content-addressing enough, or is a lightweight lock wanted for
  eviction?
- `automl data cache` verb surface + where the library function lives
  (`automl/data/` vs `automl/utils/io/`).

## 8b. Caveats / what'll bite you

Verify these against the **code** (it's the source of truth — this doc is
best-effort) before building on them:

- **The GCS client may already checksum-validate downloads.** `download_as_bytes`
  / `download_to_filename` default to checksum validation and raise
  `DataCorruption` on a bytes mismatch. So the "corrupted bytes that pass through
  to a `DataError`" we observed is *surprising* — it implies corruption **after**
  the checksum (parquet decode? `BytesIO`?), or that validation wasn't engaged on
  this client version. **Don't assume `checksum="crc32c"` is net-new behavior —
  empirically confirm what the current path does** before claiming it as the fix.
- **Two different hashes catch two different things.** crc32c is over the *bytes*;
  `dataframe_content_hash` is over the *parsed DataFrame*. Keep both — a transport
  checksum can't catch "right bytes, wrong object" or a decode bug, and the frame
  hash can't catch a truncated download as cheaply.
- **`dataframe_content_hash` itself is not free** on a 1.15M × 2,253 frame.
  Read-once removes the redundant *downloads*, but the contract builder still
  hashes each slice — measure that CPU; it may dominate once the network is gone.
- **Read-once-keep-full trades memory for reads.** The full parsed frame is large
  in RAM (10s of GB possible). Prefer "load → compute slice hashes → release",
  not "hold the full frame for the whole trial".
- **Cross-process cache root must resolve identically.** The serving-validation
  **subprocess** only hits the cache if it computes the same cache path as the
  parent — make the root absolute and env-overridable, and check the child's
  `HOME`/cwd/env (`child_env` in `serving_validation.py`) actually carries it.
- **Key on `dataset_id` + content hash, not id alone.** Content-addressing is what
  makes the cache invalidation-free; an id-only key could serve stale bytes if an
  id were ever reused.
- **Non-GCS sources must no-op.** `LocalCSVSource` (the `NEOBANK_NCM_CSV` /
  dry-run offline path) reads local files, not GCS — the cache layer has to pass
  through cleanly for non-GCS sources.
- **Concurrency is coming.** `docs/to-do/multi-runner-architecture.md` would put
  parallel trial processes against one cache dir → eviction races. Either make
  eviction concurrency-safe or document the single-writer assumption explicitly.

## 9. References

- Migrated finding: [`finding-redundant-full-reads.md`](finding-redundant-full-reads.md).
- Related (left in `to-do/`): `tiny_eval-retry-orphaned-gcs-artifact.md`
  (eval-write idempotency — different seam, same reliability theme).
- Sibling effort: [`../serving-validation-robustness/`](../serving-validation-robustness/).
- Memory: `neobank-fulldata-gcs-read-retry`, `gcs-vpn-throughput`,
  `vpn-sase-tls-ca-bundle`.
