# Trial reliability — ratified design

**Status:** ratified 2026-06-10 after a code-verification pass over `main`
(branch `core/dataset-read-reliability`, cut from `cc688b5`). This merges and
supersedes the two predecessor designs (`dataset-read-reliability`,
`serving-validation-robustness`); their findings are migrated into this folder.
Decisions below are settled with the user — don't relitigate in passing; the
remaining open items are listed in §9 and belong to the plan stage.

Implementation is split into **independently-landable plans** (one per pillar,
§3–§7), written under `plans/` next.

---

## 1. Where `main` is today (verified against code)

The read path:

- **`automl/utils/io/gcs.py:213` `read_parquet`** — `blob.download_as_bytes()`
  → `pd.read_parquet(BytesIO(...))`. Single-shot, whole-object, into RAM; no
  resumable transfer, no retry/timeout tuning passed to the client. `read_csv`
  is the same via `read_bytes`. (Installed client: google-cloud-storage 3.11.0,
  which likely already checksum-validates full downloads — see §9.)
- **`automl/mlflow/experiment/artifacts.py` `read_dataset_frame`/`read_registry`**
  — thin wrappers re-raising as `StorageError`.
- **`automl/data/registry.py:51` `load_dataset_by_id`** — reads registry CSV +
  full frame, `validate_loaded_dataset` (n_rows / n_columns / `data_content` /
  `feature_registry` / `schema` hashes vs the manifest), slices to the
  requested split, drops the full frame.
- Hashing lives in **`automl/utils/hashing.py`** (`dataframe_content_hash`,
  `schema_hash`); `data/contract.py` imports it downward. (The predecessor
  design said hashing lives in `data/contract.py` — already moot.)

**Per-trial full-frame reads (7, all in the trial-runner parent process):**

| # | Site | For |
|---|------|-----|
| 1 | `runner/trial.py:98` fit load | train split |
| 2 | `eval/_load.py:38` via `evaluate` | eval split |
| 3 | `runner/trial.py:609` `_try_log_train_eval` | train diagnostic (best-effort) |
| 4–6 | `runner/trial.py:568` `_trial_data_contract` | each non-fit split's slice hash |
| 7 | `runner/serving_validation.py:49` fixture stage | eval split again |

Every site funnels through `load_dataset_by_id` → full download + parse +
manifest verify + slice.

Serving validation (`automl/runner/serving_validation.py`):

- `subprocess.run(..., text=True, timeout=_VALIDATION_TIMEOUT_S)` with a
  hardcoded `_VALIDATION_TIMEOUT_S = 120` (line 25).
- On `TimeoutExpired`, the handler slices `exc.stderr` **but keeps it bytes**
  (line 536) — `exc.stderr` is bytes even with `text=True` — so
  `json.dumps(report)` raises `TypeError`: the "don't crash the trial" handler
  crashes the trial.
- A child that dies without writing its report (incl. signal kills) gets a
  synthesized failure report from the parent (lines 553–566) — generic, with an
  often-empty stderr tail and no signal labeling.

The loop: trials run as separate processes (`run.py` from
`automl/trial/template.py` exits 0 only on FINISHED); the agent manager skill
stops without repair on any nonzero exit (`agent-skills/skills/automl/SKILL.md`
items 11/13).

**Not on `main`:** the outer read-retry band-aid, `_decode_tail`,
`serving_validation_seconds`, `skip_live_check` — all live only on the
`neobank_ncm_v3_replicate` branch and are deliberately not ported. No disk
cache exists anywhere; this design's cache is net-new.

## 2. Corrections to the predecessor designs (load-bearing)

The 2026-06-10 verification pass found three claims that don't match `main`:

1. **The validation subprocess never reads the dataset.** The full-frame read
   is in the *parent* (fixture stage, `serving_validation.py:49`); the child
   only loads the pyfunc model + 10-row fixture files. So "the cache is the
   cross-process bridge for the subprocess" was wrong. The cache's real
   cross-process value: **each trial is its own process, and a loop of N
   trials re-downloads the same immutable dataset N×7 times** — the cache
   collapses that to one populate per dataset per machine.
2. **A child native crash is not silent.** The parent already synthesizes a
   failure report + `validation.status` tag when the report file is missing.
   What *is* silent and halting is the **runner parent process** dying
   natively (the observed exit 139 — no MLflow finalization, no error tag; the
   fixture-stage torch predict runs in the parent and is a plausible culprit).
   In-process code cannot catch that; visibility comes from crash-safe local
   records (§6) + `faulthandler`, and the coder/skill layer already reports
   the exit code.
3. **`main` is already fail-soft for every validation outcome** — nothing in
   `_run_trial` checks the validation report's status; timeout (post decode
   fix), child crash, and even a correctness mismatch all leave the trial
   FINISHED with a `validation.status=failed` tag. The loop halted in the
   field only because *exceptions* escaped (the bytes `TypeError`; transient
   `StorageError` on one of the 7 reads).

## 3. Settled policy: fail-soft, with deployability as a tag

Ratified with the user:

- **Any trial that runs to completion is FINISHED.** No serving-validation
  outcome (correctness, latency, timeout, child crash) changes trial status or
  halts the loop. `validation.status` is the deployability signal — a model
  with a failed validation is "not servable", visible on the run, and that is
  enough.
- **The fix for infra failures is visibility, not policy** — accumulate and
  publish errors (§6) so a FINISHED trial still reports everything that went
  wrong inside it.
- **Retry/modular runner is explicitly deferred** to a future effort; this
  design only lays its foundation (§6, record vs machinery).

**Known boundaries (consciously accepted, 2026-06-10):**

1. **The loop still halts on the first hard-failed trial.** Fail-soft covers
   only trials that complete; a thrown exception still exits 1 and the manager
   skill stops without repair. This effort shrinks the probability, not the
   blast radius — loop-survival policy belongs to the future retry effort.
2. **A correctness-failed model stays FINISHED on the leaderboard.** Anything
   that selects "the best model" for export/serving must respect
   `validation.status`; the plans include an audit of leaderboard consumers.
3. **A parent native crash is diagnosable, not durably recorded** — the MLflow
   run stays RUNNING; evidence lives in stderr + the local issues JSONL.
   Closing this needs a supervisor seam outside the runner process, parked at
   `docs/to-do/runner-crash-supervision.md`.

## 4. Pillar 1 — content-addressed dataset cache

```
load_dataset_by_id                                   ← data read seam (policy)
  └─ blob_cache.get_or_populate(key, populate_fn)    ← utils/io (generic mechanism)
        ├─ HIT  → local parquet (no network)
        └─ MISS → robust download (§5) → atomic tmp→rename → verify
automl data cache {list,prune,clear}                 ← thin CLI verb
```

- **Generic mechanism → `automl/utils/io/blob_cache.py` (leaf).** Key → file;
  atomic writes (download to tmp, rename into place); **LRU-on-write** eviction
  under a size cap; touch-on-read for recency. Knows nothing about datasets —
  reusable for any heavy artifact later.
- **Dataset policy → `data/registry.load_dataset_by_id`**, the one place with
  the `Dataset` record in hand. Key = `dataset.id` + 
  `component_hashes.data_content` (content-addressing keeps the cache
  invalidation-free; an id-only key could serve stale bytes). Both the heavy
  `data.parquet` **and** the small `feature_registry.csv` are cached under the
  same key (near-free, removes the second network read).
- **Store the raw parquet file, parse on read.** Parse cost remains but is
  local; revisit only if measurement shows parse dominating (§9).
- **`validate_loaded_dataset` stays** as defense in depth: every load — hit or
  miss — still verifies the parsed frame against the manifest hashes. GCS
  remains the single source of truth.
- **Retention:** cache root **`~/.cache/brigit-automl/`** (never in the repo
  or `projects/`), overridable via **`AUTOML_CACHE_DIR`**; resolve to an
  absolute path so child processes inherit the same root. Default size cap
  **20 GB**. No TTL — datasets are immutable and a run pins one; LRU recency
  is the only eviction signal.
- **Concurrency:** atomic rename + content-addressed keys make readers safe
  against concurrent populates (a reader sees a complete file or misses and
  re-downloads). Eviction assumes a **single writer** for now — documented,
  not locked; the `multi-runner-architecture` to-do owns the rest.
- **Non-GCS sources bypass cleanly** (`LocalCSVSource` / local URIs read local
  files — the cache layer must be a pass-through for them).
- **CLI:** `automl data cache list|prune|clear` in `automl/cli/`, thin wrappers
  over a library function (surfaces stay thin).

## 5. Pillar 2 — robust populate + read-once contract builder

- **Populate (cache miss)** uses the storage client's native robustness
  instead of bare `download_as_bytes()`:
  `blob.download_to_filename(tmp, checksum=..., retry=<tuned>)` — streamed to
  disk (no bytes+frame double-residency), chunked/resumable transfer,
  checksum validation, and a retry predicate/deadline tuned to also cover the
  oauth2 token-refresh timeouts observed in the field. **First plan step:
  empirically confirm what client 3.11.0 already validates by default** —
  don't claim the checksum as net-new behavior without evidence (the observed
  "corrupted bytes reaching the manifest check" implies the corruption may be
  post-checksum, e.g. decode-side).
- **Two hashes, two jobs — keep both:** crc32c covers the transported bytes;
  `dataframe_content_hash` covers the parsed frame ("right bytes, wrong
  object", decode bugs). Neither subsumes the other.
- **Read-once contract builder:** `_trial_data_contract` stops re-reading per
  split. One full-frame load (a cache hit), mask each split predicate in
  memory, compute each slice's `n_rows` + `dataframe_content_hash`, release
  the frame. **Slice-hash semantics identical** (hash the sliced frame) so
  existing `SliceContract`/tag-lineage verification is unchanged. One parse
  instead of three; no full frame held across the trial. Do **not** thread a
  full-frame object through the eval domain — with the cache, the eval and
  diagnostic re-reads are already cheap local reads.
- Nothing to retire on `main`; the neobank branch's `_read_and_verify_dataset`
  retry drops out at rebase once this lands.

## 6. Pillar 3 — serving-validation hardening

- **Decode fix:** stderr/stdout tails decode bytes
  (`raw[-1000:].decode("utf-8", "replace")`), with a regression test feeding
  `TimeoutExpired(stderr=b"...")` through the handler and asserting the report
  serializes.
- **Configurable timeout:** `serving_validation_seconds` as a `RunConfig`
  field (operational knob, like `per_trial_seconds` — which remains advisory
  and unenforced, a separate to-do; only this one is a real
  `subprocess.run(timeout=)`). **Default 300s** — the observed baseline sat at
  120.07s, so the old default is boundary-tight by evidence. Timeout error
  message references the configured value.
- **Signal-exit guard:** on `completed.returncode < 0`, synthesize a *labeled*
  report (`error_class: "SignalExit"`, signal number + name, e.g. SIGSEGV/-11)
  and record a ledger issue (§7) — replacing today's generic
  missing-report fallback for that case.
- Trial status: unchanged (fail-soft per §3).

## 7. Pillar 4 — `TrialContext` + issue ledger

The organizing principle, agreed with the user: **separate the record from the
machinery.** The record (timings, issues — schema'd, published) must be stable;
the machinery (straight-line script today, composable retryable steps someday)
is the future effort's to redesign. Invest in the record now; the future
step-driver inherits it untouched.

- **`TrialContext`** (new, `automl/runner/`) composes the trial's cross-cutting
  state: identity (run_id, trial_id, slug, strategy, trial_number — filled in
  as they become known), the existing `TimingRecorder`, the new
  `IssueRecorder`, and the trial dir. One object threads through the runner
  instead of N parameters (`_publish_failure_artifacts`'s ~10 params collapse
  into it). `with ctx.phase(...)` keeps today's timing semantics —
  `TimingRecorder` survives unchanged inside it (composition, not rewrite).
- **`IssueRecorder`:** `ctx.record_issue(exc_or_message, severity=...)`
  captures `{phase, severity, error_class, message, traceback_tail, at}`
  (reusing `ExceptionSnapshot` from `runner/failures.py`). Events **append to
  a local JSONL in the trial dir as they happen** — crash-safe: a parent
  SIGSEGV still leaves evidence on disk. At trial end — success *and* failure
  paths — published to MLflow as `trial/issues.json` plus a
  **`trial.issue_count`** tag so the leaderboard/agent can see at a glance
  that a FINISHED trial had problems.
- **Silent sites convert** to `ctx.record_issue(...)` instead of swallowing:
  `_try_log_train_eval`'s bare `except` (the lived example — train-set eval
  failures we never see), the validation timeout path, the signal-exit path,
  latency `not_measured`.
- **`faulthandler.enable()` at runner entry** so a native crash of the runner
  process dumps a Python-level traceback to stderr before dying (pairs with
  the coder layer already reporting the exit code).
- **Out of scope, by prior decisions:** the general logging/observability pass
  (P3, parked 2026-05-29 — this is a trial-scoped ledger, not a logging
  framework) and retry/modularization (future effort; needs its own design for
  step idempotency and resume semantics).

## 8. Companion: `skip_snowflake_live_check` on `RunConfig`

- Flat boolean field, default `False`, on `RunConfig` (operational knob,
  consistent with `serving_validation_seconds`). `validate_project`'s Snowflake
  connectivity check reads it: env-var and SQL-file checks still run; only the
  live `SELECT 1` probe is gated, with the same skip warning. The
  `probe_snowflake` CLI/test override stays and wins over config.
- Verified: `RunConfig` does not enter dataset identity (identity comes from
  the source's `recipe_identity`) — the flag can never change a dataset hash.
  Confirm once more in the plan when wiring.
- Follow-through at neobank rebase: `projects/neobank_ncm/config.py` moves off
  `SnowflakeSource(skip_live_check=True)` to the new field; the source flag is
  never created on `main`.

## 9. Open items for the plan stage (not design questions)

- **Empirical GCS checksum check** (§5): what does 3.11.0 validate by default
  on `download_as_bytes` vs `download_to_filename`? Decides whether the
  populate's checksum is net-new or just made explicit.
- **Slice-hash CPU**: with the network gone, measure `dataframe_content_hash`
  on the 1.15M × 2,253 slices — if hashing dominates the contract step, note
  it (optimization is out of scope here).
- **Ledger location when `trial_dir` is None** (project-run trials without a
  trial folder): pick a temp/session-scoped fallback for the JSONL.
- Exact `issues.json` schema fields and the tag name (`trial.issue_count`
  proposed); exact cache CLI output shapes.
- Retry tuning specifics for the populate (predicate, deadline values).

## 10. Testing

- **Unit:** blob cache (hit / miss / atomic rename / LRU eviction at cap /
  touch-on-read), decode regression (`TimeoutExpired(stderr=b"...")`
  serializes), signal-exit guard (child killed by signal → labeled report),
  ledger (publish on success and failure paths; JSONL survives an aborted
  run), `RunConfig` new fields' validation.
- **Contracts:** new CLI verbs (`automl data cache ...`) and any pinned doc
  phrases — change the shape, update the matching contract in the same change.
- **Integration:** file-backed MLflow trial run with a seeded cache (hit path)
  and an empty cache (populate path, local file:// or fake client).
- **E2E:** `projects/example_homecredit/` unchanged, under a `qa/` namespace.

## 11. References

- Migrated findings (this folder): `finding-redundant-full-reads.md`,
  `finding-timeout-crash-and-halt.md`, `finding-nn-pyfunc-sigsegv.md`.
- Superseded predecessor efforts: `dataset-read-reliability/`,
  `serving-validation-robustness/` (removed from `execution/` when this
  merged design landed; history in git).
- Parked out of this design: `to-do/runner-crash-supervision.md` (supervisor
  seam for natively-crashed runner processes — boundary #3 in §3).
- Related, left in `to-do/`: `tiny_eval-retry-orphaned-gcs-artifact.md`
  (eval-write idempotency — different seam, same theme),
  `tiny_per-trial-seconds-not-enforced.md` (decided: docs fix only),
  `multi-runner-architecture.md` (owns cache eviction concurrency),
  `logging-and-observability.md` (P3; owns the general logging design),
  `loop-observability.md` (stdout narration wants).
- Memory: `neobank-fulldata-gcs-read-retry`, `gcs-vpn-throughput`,
  `vpn-sase-tls-ca-bundle`, `automl-agent-logging-decisions`.
