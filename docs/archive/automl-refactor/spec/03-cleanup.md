# Cleanup Orchestration — Sub-Spec Design

**Date:** 2026-05-22
**Parent spec:** `00-structural-design.md` §15.1 Priority 3, §8.1
**Status:** Design approved; ready to inform implementation.
**Scope:** The cascading delete operation across MLflow records, GCS blobs, and local files at three granularities (project / experiment / trial).

This sub-spec defines the **delete cascade** — what gets removed at each scope, in what order, with what atomicity, and how dry-run / confirmation are handled. The CLI shape that exposes this cascade was settled during this sub-spec's first design pass; see §1 below for the carry-back to the structural spec.

---

## 1. CLI shape — carry-back to structural spec §11.1 (SETTLED 2026-05-22)

Before designing the cascade, this sub-spec triggered a holistic CLI audit and a philosophy decision that affects the whole CLI surface, not just cleanup. The decision is captured in the structural spec's updated §11.1; this section records the rationale.

### 1.1 The decision

**Cleanup as a single cross-cutting verb (`automl cleanup --scope X --name Y`) is gone.** It splits into three sibling noun-action verbs:

```
automl [--dry-run] project delete <name> [--apply] [--hard-delete] [--json]
automl [--dry-run] experiment delete <id> [--apply] [--hard-delete] [--json]
automl [--dry-run] trial delete <run_id> [--apply] [--hard-delete] [--json]
```

`--dry-run` is the top-level session flag (§4); `--apply` is the destructive opt-in (§5); `--hard-delete` opts into MLflow backend gc (§6.4); `--json` is for machine-readable output.

This is part of a broader CLI realignment: **noun-first across the board, with `automl validate <target>` as the one cross-cutting carve-out** (because `validate/` is a framework layer, not a domain). See structural spec §11.1 for the full 21-verb catalog and the legacy→new mapping.

### 1.2 Why a CLI realignment was the right move here

The legacy CLI mixed four patterns (noun-action, action-with-target-flag, action-with-positional, top-level action), with cleanup being the worst-flag-soup offender. A holistic audit surfaced six concrete inconsistencies — asymmetric lifecycle verbs (`experiment create` without `experiment delete`), duplicate surfaces (`proposal validate` + `validate proposal`), `inspect` masquerading as a noun, the `Session` naming clash with `session lock`, the unprefixed top-level `automl run`, and cleanup's six-flag delete-one-trial command. Settling philosophy now (rather than after sub-spec 09) means every subsequent sub-spec knows what CLI shape its domain produces.

### 1.3 Library implications for cleanup

The legacy `project.cleanup(scope=..., name=..., ...)` single function becomes three sibling functions on the noun whose lifecycle they end:

| CLI verb | Library function | Lives in |
|---|---|---|
| `automl project delete <name>` | `project.delete(name, *, ...)` | `project/cleanup.py` |
| `automl experiment delete <id>` | `experiment.delete(experiment_id, *, ...)` | `experiment/cleanup.py` (thin) → `project/cleanup.py` cascade infra |
| `automl trial delete <run_id>` | `trial.cleanup.delete(run_id, *, ...)` | `trial/cleanup.py` (thin) → `project/cleanup.py` cascade infra |

Shared cascade infrastructure (target enumeration, ordered delete loop, atomicity helpers, result aggregation) lives in `project/cleanup.py` — that file is the cascade engine. The per-noun functions are small wrappers that fix the scope and translate the noun's identifier into the cascade engine's input.

This preserves the structural spec §8.1 rule ("the cleanup cascade starts from Project") — the engine is in `project/`, the noun-shaped entry points are in their respective domains.

### 1.4 Verbs added during this audit

The catalog gained `list` verbs per the noun-action symmetry rule (if you can create/delete a thing, you can list things). Specifically: `project list`, `experiment list`, `trial list`, `data list`, `eval list`. These were sketched but not in legacy; the structural spec §11.1 now lists them with their library destinations.

---

## 2. The cascade — design tree status

- §3 Target set per scope — **SETTLED 2026-05-22**
- §4 dry_run handling — **SETTLED 2026-05-22**
- §5 Plan vs apply pattern — **SETTLED 2026-05-22**
- §6 Delete ordering — **SETTLED 2026-05-22**
- §7 Atomicity + error model — **SETTLED 2026-05-22**
- §8 Confirmation mechanism — **SETTLED 2026-05-22**
- §9 Library function signatures — **SETTLED 2026-05-22**
- §10 Result shape — **SETTLED 2026-05-22**
- §11 Orphan handling — **SETTLED 2026-05-22**
- §12 What this sub-spec defers

---

## 3. Target set per scope (SETTLED 2026-05-22)

The cascade deletes **artifacts**, never user-authored code. Each scope has three layers (MLflow, GCS, local files); cleanup at any scope covers all three.

**One invocation cleans one universe.** Cleanup respects `session.dry_run` strictly: the values in the tables below are written using a `<route>` placeholder defined by session mode. To clean the other universe, run the command again with the opposite `--dry-run` setting.

### Routing convention

Throughout this sub-spec, `<route>` is the namespace-, mode-, and-scope-aware path prefix derived from the session:

| Scope | `<route>` when `session.dry_run=False` | `<route>` when `session.dry_run=True` |
|---|---|---|
| project | `<project>` | `dry_run/<project>` |
| experiment | `<project>/<id>` | `dry_run/<project>/<id>` |
| trial | parent experiment's `<route>` (looked up from `run_id`) | same |

**Namespace prefix.** When `session.namespace` is non-empty (the top-level `--namespace <name>` flag, sub-spec 01), every `<route>` above is additionally prefixed by `<name>/` — i.e. the full segment order is `[<namespace>/][dry_run/]<project>[/<id>]`. So `automl --namespace qa experiment delete <id> --apply` cleans only `qa/<project>/<id>` (or `qa/dry_run/<project>/<id>` with `--dry-run`), never the real (`""`) namespace. The default `""` reproduces today's routes exactly.

Every path in the tables that follows uses `<route>` — there is no "and also the dry_run twin" (or "and also the other namespace") branch anywhere. One invocation = one (namespace, mode) universe.

### Per-scope target table

| Scope | MLflow | GCS | Local files | NOT deleted |
|---|---|---|---|---|
| **`project delete <project>`** | `<route>/overview` experiment + every experiment whose name starts with `<route>/` (cascades to all child runs) | All blobs under `gs://<bucket>/<gcs_prefix>/<route>/` | `.cache/automl/tmp/{proposals,timelines,session_locks}/<route>/...`; framework-generated trial sandbox dirs under `projects/<project>/experiments/<route>/...` | **`projects/<project>/` user-authored code** (config.py, data/pipeline.py overrides, PROJECT_INSTRUCTIONS.md, SQL files); other projects; the opposite-mode universe |
| **`experiment delete <id>`** | One MLflow experiment named `<route>` + all its runs (cascade) | Blobs under `gs://<bucket>/<gcs_prefix>/<route>/` | `.cache/automl/tmp/{proposals,timelines,session_locks}/<route>/`; framework-generated trial sandbox dirs under `projects/<project>/experiments/<route>/` | Project overview; sibling experiments under the same project; user code; the opposite-mode universe |
| **`trial delete <run_id>`** | The one MLflow run | Blobs under that run's `run_bulk` GCS prefix | The trial sandbox dir under `projects/<project>/experiments/<route>/<trial_name>/`; per-trial session-lock dir if held | Parent MLflow experiment; sibling trials; experiment-overview tags (incl. any `active_dataset_hash` pin — see §3.2) |

### 3.1 Three load-bearing exclusions

These came up explicitly during design and are written down so they don't get re-litigated:

1. **`projects/<name>/` recipe folder** — never deleted. The user wrote that code. Project-delete only removes framework-generated artifacts. The clean separation: `projects/<name>/config.py`, `projects/<name>/data/pipeline.py`, `projects/<name>/PROJECT_INSTRUCTIONS.md`, and SQL files in `projects/<name>/sql/` are user-authored; `projects/<name>/experiments/<trial_name>/` directories are framework-generated trial sandboxes and ARE deleted.

2. **`automl_runs/`** — out of scope entirely. This is the parent workspace's test-harness folder (peer to `automl_dev/`, holds per-iteration working copies like `homecredit-001/`). The library has no functional dependency on it (only `pyrightconfig.json` excludes it from type-checking). Cleanup never touches it.

3. **Other projects under `projects/`** — never touched by single-project delete. Cleanup is strictly scoped to the named project.

**`dataset_index.json` deletion is intentional.** Project-delete wipes everything under the project-overview MLflow run, which includes the `dataset_index.json` artifact (the project's Dataset registry). This is correct — when the project is being torn down, the Dataset registry goes with it. If the user later re-creates the project via `automl project init <name>`, they get a clean slate (the recipe code remains, but no stale registry references).

### 3.2 The `active_dataset_hash` pin survives trial-delete

`mlflow.experiment.set_active_dataset(snapshot_name)` writes a tag on the experiment-overview run pointing at a project-level Dataset. Datasets are *project-level* (shared across experiments by content hash, registered in `project_overview/dataset_index.json`), not trial-level. Trials reference Datasets via their own `automl.trial.dataset_hash` tag, but deleting a trial doesn't invalidate the experiment's active-dataset pin — the pin still points at a Dataset that still exists.

Cleanup does NOT mutate experiment-overview tags during trial-delete. This keeps trial-delete a per-trial operation that doesn't reach up to mutate parent state.

### 3.3 dry_run and real are strictly isolated universes (SETTLED 2026-05-22)

The two run modes route to **completely separate paths at every layer**:

| Layer | Real path | Dry-run path |
|---|---|---|
| MLflow experiment names | `<project>/<id>` | `dry_run/<project>/<id>` |
| GCS prefix | `gs://.../<gcs_prefix>/<project>/...` | `gs://.../<gcs_prefix>/dry_run/<project>/...` |
| Local cache `.cache/automl/tmp/{proposals,timelines,session_locks}/` | `.../<project>/...` | `.../dry_run/<project>/...` |
| Trial sandbox dirs | `projects/<project>/experiments/<project>/<id>/<trial_name>/` | `projects/<project>/experiments/dry_run/<project>/<id>/<trial_name>/` |

**Every layer segregates by path prefix.** No metadata-content filtering. The legacy already does this for the first three layers (via `_project_route()`); the new design extends it to trial sandbox dirs too — see §3.4 below.

**Cleanup never crosses the mode boundary.** Delete-dry_run touches only dry-run-mode artifacts; delete-real touches only real-mode artifacts. Two separate operations.

### 3.4 Trial sandbox dirs — path-segregated (CARRY-BACK to sub-spec 08)

Legacy co-locates real and dry-run trial dirs under `projects/<project>/experiments/<trial_name>/` and filters by `metadata.json::run_mode` to distinguish modes. **The new design segregates these by path prefix**, matching every other layer:

```
projects/<project>/experiments/<project>/<id>/<trial_name>/           # real
projects/<project>/experiments/dry_run/<project>/<id>/<trial_name>/   # dry_run
```

This makes cleanup a single tree-delete (`rm -rf projects/<project>/experiments/dry_run/<project>/<id>/`) instead of an iterate-and-filter. Promotion of a trial from dry-run to real (if/when that workflow exists) becomes a move between siblings — no metadata juggling.

**Owner: sub-spec 08 (Runner).** The runner creates sandbox dirs; this decision tells the runner where. Sub-spec 03 records the cleanup-side requirement; sub-spec 08 implements it. Added to `open-questions.md` as a carry-back.

---

## 4. dry_run handling — top-level `--dry-run` flag, session-wide (SETTLED 2026-05-22)

### 4.1 The model

`--dry-run` is a **top-level flag on `automl` itself, before any verb** — it's an environment switch, not a per-verb argument. Present → all verbs route to the dry_run universe (different MLflow namespace, different GCS prefix, possibly limited data). Absent → real universe. Maps directly to `Session.dry_run: bool` in sub-spec 01.

The CLI dispatcher reads `--dry-run` at the top level and threads it into `use_project(..., dry_run=...)`. Verb subparsers don't see the flag.

```bash
# dry_run universe
automl --dry-run experiment delete X --apply
automl --dry-run experiment run
automl --dry-run trial delete <run_id> --apply
automl --dry-run data profile

# real universe (default)
automl experiment delete X --apply
automl experiment run
```

This is **structural-spec §11.1 territory** (carry-back applied 2026-05-22 — see new "Top-level flags" subsection there), not cleanup-specific. Every stateful verb works the same way.

### 4.2 What's NOT in the verb's argument list

- **No `--mode` flag.** Mode is session state, set by top-level `--dry-run`.
- **No `--dry-run / --real` pair.** Single boolean — present or absent.
- **No `--yes` flag.** Destructive opt-in is `--apply` (§5); preview-by-default is the safety net.
- **No `--both` shortcut.** Universes are strict-isolated (§3.3); to clean both, run twice.

### 4.3 Trial-delete consistency rule

`automl trial delete <run_id>` operates only on trials within the session's mode. If the `run_id` belongs to the other mode (e.g., user passes a real-mode `run_id` while in a dry_run session), the command errors out:

```
ERROR: run_id <id> not found in current session's mode.
       Did you mean to use --dry-run (or omit it)?
```

This keeps the universal rule "session.dry_run dictates which container any verb operates on" with zero carve-outs. The user is always operating in exactly one universe at a time.

### 4.4 Library-level signatures (no dry_run parameter)

All three delete functions follow the same shape — `<identifier>` + `apply` + `session`. Mode comes from `session.dry_run` (sub-spec 01 convention):

```python
# project/cleanup.py
def delete(name: str, *, apply: bool = False, session: Session | None = None) -> CleanupReport: ...

# experiment/cleanup.py
def delete(experiment_id: str, *, apply: bool = False, session: Session | None = None) -> CleanupReport: ...

# trial/cleanup.py
def delete(run_id: str, *, apply: bool = False, session: Session | None = None) -> CleanupReport: ...
```

Programmatic callers set mode via session (`automl.use_project(name, dry_run=True)` once at session start, then call `delete()` freely). The `dry_run` value is part of session, not a per-call argument — matching every other Tier 2 function in the library.

### 4.5 Layer-selective cleanup (deferred — not in v1)

If layer-selective cleanup ever becomes necessary, the shape is **a single `--only-layers` flag with comma-separated values**, not three separate flags. Default (omitted) = all three layers. Not shipped in v1 — recorded only so a future addition doesn't fragment into separate `--mlflow-only` / `--gcs-only` / `--local-only` flags.

---

## 5. Plan / apply pattern (SETTLED 2026-05-22)

### 5.1 Two-phase: preview by default, `--apply` to commit

Every `delete()` call goes through two internal phases:

1. **Plan** — read MLflow + scan GCS + walk local filesystem; build a `CleanupPlan` describing what would be deleted. No mutations.
2. **Apply** — execute the deletes in the order from §6. Only runs when `apply=True`.

The same library function handles both via the `apply: bool = False` keyword arg. Default is `False` → plan-only → returns the plan in `CleanupReport.plan`, exits 0, deletes nothing. Pass `apply=True` to commit.

CLI mirror: omit `--apply` for preview; pass `--apply` to destroy.

### 5.2 `--apply` is the only destructive opt-in

No `--yes`. No interactive prompt. No `--confirm-name`. The verb's preview-by-default plus an explicit `--apply` IS the confirmation flow.

Reasoning:
- **Agent-friendly**: humans and agents both use `--apply` identically. No TTY-bound prompts to special-case.
- **Preview-first workflow**: typical use is `delete X` → review output → `delete X --apply`. Two-step interaction is the natural human safety net; `--apply` itself is the opt-in.
- **Flag minimalism**: per the user's explicit preference, the verb stays at two flags max (`--apply` and the optional `--hard-delete` from §6.4).

If a "type the project name to confirm" guard becomes necessary later (e.g., production deployments where catastrophic-project-delete is a real risk), add `--confirm-name=<name>` then. Follow demand.

---

## 6. Delete ordering (SETTLED 2026-05-22)

### 6.1 Layer order: MLflow → GCS → local

The natural inverse of sub-spec 02 §3.5's writer order (GCS-then-MLflow). MLflow is the ledger; un-commit it first.

| Step | Layer | Operation |
|---|---|---|
| 1 | MLflow | Soft-delete experiment(s) / run(s). MLflow cascades the deletion to child runs automatically — no explicit per-run iteration needed. |
| 2 | GCS | List + delete every blob under the relevant prefix(es). Uses `_atomic.py` from sub-spec 02 §7 for the batch. |
| 3 | Local | `shutil.rmtree(...)` the relevant local directories. |

Failure-mode reasoning:

| Failure point | State left | Recoverable? |
|---|---|---|
| Step 1 (MLflow) | Nothing deleted | Yes — re-run cleanup |
| Step 2 (GCS) after Step 1 | MLflow soft-deleted; orphan GCS blobs | Yes — re-run cleanup; orphan blobs caught by prefix-scan (§11) |
| Step 3 (local) after Steps 1+2 | MLflow + GCS gone; orphan local dirs | Yes — re-run or just `rm -rf` manually |

The MLflow-first ordering ensures partial failures leave only orphaned bytes (recoverable), never broken references (a run in MLflow pointing at empty GCS, which would crash readers).

### 6.2 Per-scope concrete ordering

Uses the `<route>` placeholder from §3 (always one universe per invocation):

| Scope | Step 1 — MLflow | Step 2 — GCS | Step 3 — local |
|---|---|---|---|
| **trial** | `delete_run(run_id)` (the one run) | Delete blobs under the run's `run_bulk` prefix (computed from parent `<route>` + run_id) | `rm -rf projects/<project>/experiments/<route>/<trial_name>/` + per-trial session-lock dir |
| **experiment** | `delete_experiment(<route>)` (cascades to its runs) | Delete blobs under `gs://<bucket>/<gcs_prefix>/<route>/` | `rm -rf .cache/automl/tmp/{proposals,timelines,session_locks}/<route>/` + `rm -rf projects/<project>/experiments/<route>/` |
| **project** | `delete_experiment(<route>/overview)` + soft-delete every experiment whose name starts with `<route>/` | Delete blobs under `gs://<bucket>/<gcs_prefix>/<route>/` | `rm -rf .cache/automl/tmp/{proposals,timelines,session_locks}/<route>/` + `rm -rf projects/<project>/experiments/<route>/` |

The single `<route>` substitution covers both modes — `dry_run/...` paths are reached by `automl --dry-run <verb>`, the real paths by omitting `--dry-run`. No invocation ever touches the opposite universe.

### 6.3 Soft delete is the default; plan uses `view_type=ACTIVE_ONLY`

MLflow's `delete_experiment` and `delete_run` are **soft deletes** — items move to `lifecycle_stage="deleted"`; they're hidden from default listings (`list_experiments` / `search_runs` with `view_type=ACTIVE_ONLY`) but linger in backend storage. From every user-facing surface, a soft-deleted experiment is gone.

**Plan enumeration uses `view_type=ACTIVE_ONLY` — already-soft-deleted experiments are NOT re-enumerated.** Rationale: an already-deleted record was deleted for a reason; cleanup shouldn't resurrect it just to re-attempt a downstream operation. Concrete consequence: if a `--hard-delete` invocation soft-deletes successfully but the subsequent `mlflow gc` step fails, a follow-up `--hard-delete` invocation will NOT find the now-soft-deleted experiment and so will not re-attempt the gc. Those soft-deleted records become MLflow-backend lint to be cleaned by a separate admin tool (`automl mlflow gc` if/when added). Acceptable trade-off — the alternative ("re-enumerate everything including soft-deleted") would mean cleanup acts on items the user already said "delete this" about.

Hard delete (permanent removal from MLflow backend storage) requires running `mlflow gc` against the backend store, which needs direct backend access (hosted MLflow tracking APIs only expose soft delete). Available via opt-in flag — see §6.4.

For project-scope enumeration, cleanup calls `mlflow.project.list_experiments()` (sub-spec 02 §6.1) — which internally filters to `view_type=ACTIVE_ONLY` matching the `<name>/*` prefix.

### 6.4 `--hard-delete` flag for permanent MLflow removal

When the user wants to reclaim MLflow backend storage in addition to clearing the soft-deleted records, they pass `--hard-delete`:

```bash
automl experiment delete X --apply                  # soft delete only (default)
automl experiment delete X --apply --hard-delete    # soft delete + mlflow gc on backend
```

Library: `hard_delete: bool = False` keyword arg on all three `delete()` functions. When True:
- Step 1 still does the soft delete first (MLflow's required two-stage process).
- After Steps 1–3 complete, run `mlflow gc --backend-store-uri <uri> --experiment-ids <ids>` to permanently remove the records. Experiment IDs come from `CleanupResult.mlflow_experiment_ids` (populated during the soft-delete step in Step 1).
- **Artifacts destination**: when `session.config.mlflow_artifacts_destination` is non-empty (read from `MLFLOW_ARTIFACTS_DESTINATION` env var per sub-spec 01 §3.1), pass it as `--artifacts-destination <value>` to the `mlflow gc` subprocess. Required when GCS is configured as MLflow's artifact store; omitting it would cause gc to miss artifact files. Matches legacy lines 597-602.
- Backend store URI resolution: `MLFLOW_BACKEND_STORE_URI` env var, or auto-detect from local SQLite file (`mlflow_local/mlflow.db`), or fail with a clear error directing the user to provide it.
- Hosted MLflow without direct backend access → `--hard-delete` raises `StorageError` with an actionable message ("hosted MLflow tracking does not expose backend store; --hard-delete requires direct access").

Naming: explicit ("hard-delete" is unambiguous vs the soft-delete default). Replaces legacy's `--purge-mlflow` which was opaque.

---

## 7. Atomicity + error model (SETTLED 2026-05-22)

### 7.1 Continue-and-collect within the cascade

Within a single `delete()` invocation, the cascade goes through every target in plan order. **Per-target failures are recorded but don't halt the cascade.** Each layer's targets get individual status:

| Status | Meaning |
|---|---|
| `"deleted"` | Target was present; now gone. |
| `"skipped: not found"` | Target wasn't there to begin with. Not an error. |
| `"skipped: already deleted"` | Target was already in a deleted lifecycle stage (MLflow). Not an error. |
| `"failed: <reason>"` | Operation was attempted and the backend rejected it. Recorded; cascade continues. |

Result lands in `CleanupResult` (§10) — caller sees the full picture in one pass.

### 7.2 When the cascade halts (raises StorageError)

Halting is reserved for *systemic* failures, not per-target ones:

- MLflow tracking server unreachable (cannot enumerate at all)
- GCS bucket access denied entirely (cannot list any prefix)
- Local filesystem write-protected (cannot `rmtree` any path)

These raise `StorageError` (sub-spec 02 §11) from the seam, halting the cascade. The caller's report shows whatever completed before the failure.

### 7.3 Idempotent re-runs

Cleanup is idempotent. Re-running `delete X --apply` after a partial failure:
- MLflow targets already in `deleted` state → "skipped: already deleted"
- GCS prefixes with nothing under them → "deleted 0 blobs"
- Local paths that don't exist → "skipped: not found"

So the recovery loop is just "fix the underlying issue (auth, network), re-run." No special recovery mode.

### 7.4 Connection to sub-spec 02's `_atomic.py`

The batch-GCS-delete step uses `_atomic.py` (sub-spec 02 §7) — list + delete in a transactional helper that handles partial failure by recording each blob's delete status. If `_atomic` is enhanced to support delete-with-rollback (not currently planned), cleanup picks it up automatically.

### 7.5 Per-blob GCS error handling (implementation requirement)

The legacy `_delete_gcs_prefix` (line 788-791) loops over blobs and calls `blob.delete()` with no per-blob try/except — a single permission error aborts the prefix delete and propagates up. The new continue-and-collect model REQUIRES the implementation to wrap each individual blob delete in `try / except` so per-blob failures collect into the result rather than abort the cascade. The aggregated count + per-failed-blob status lands in `CleanupResult.gcs[<prefix>]`. This is a behavioral change from legacy; flagged here so implementers don't accidentally inherit legacy's fail-fast pattern.

### 7.6 MLflow `delete_run` lifecycle pre-check

Before calling `client.delete_run(run_id)`, the implementation calls `client.get_run(run_id)` and inspects `run.info.lifecycle_stage`:
- `"active"` → call `delete_run`, record `"deleted"`
- `"deleted"` → skip the call, record `"skipped: already deleted"`
- `get_run` raises → record `"skipped: not found"`

This gives `CleanupResult` accurate per-run status distinctions. Same pattern for `delete_experiment` (use `client.get_experiment_by_name(name)` and inspect lifecycle).

### 7.7 Concurrent delete operations are not supported within one process

`automl.session()` reads a process-level contextvar; the `mlflow.bind()` state is also process-level (sub-spec 02 §5). Two delete operations running concurrently in the same process (e.g., async tasks) would race over the session/bind state and produce undefined behavior.

Document the constraint: cleanup is designed for single-operation-at-a-time use. CLI invocations are inherently serial (one process per invocation). Programmatic callers running multiple deletes should run them sequentially in one process or one per subprocess. Cross-process concurrent deletes are fine (separate contextvars per process).

### 7.8 Empty cleanup case

If a `delete()` call's plan contains zero targets at every layer (project doesn't exist, no GCS blobs, no local dirs), the cascade is a no-op: returns `CleanupReport(plan=<empty>, applied=apply_flag, result=<empty if applied>)`. Exit code 0. No warning, no error — the operation is idempotent and "delete a thing that isn't there" is a successful no-op by definition.

---

## 8. Confirmation mechanism (SETTLED 2026-05-22)

Resolved in §5.2: **`--apply` is the only destructive gate**. No `--yes`, no `--confirm-name`, no interactive prompt. Preview-by-default + explicit `--apply` is the confirmation flow. Section retained as a header for cross-reference; full text in §5.2.

If catastrophic-project-delete safety ever becomes a real concern, add `--confirm-name=<name>` then (follow demand). Not in v1.

---

## 9. Library function signatures (SETTLED 2026-05-22)

Three sibling functions with identical shape:

```python
# project/cleanup.py
def delete(
    name: str,
    *,
    apply: bool = False,
    hard_delete: bool = False,
    session: Session | None = None,
) -> CleanupReport: ...

# experiment/cleanup.py
def delete(
    experiment_id: str,
    *,
    apply: bool = False,
    hard_delete: bool = False,
    session: Session | None = None,
) -> CleanupReport: ...

# trial/cleanup.py
def delete(
    run_id: str,
    *,
    apply: bool = False,
    hard_delete: bool = False,
    session: Session | None = None,
) -> CleanupReport: ...
```

Each is a small wrapper (~10–20 lines) over the shared cascade engine in `project/cleanup.py`. The engine handles:
- Plan construction (`_build_plan(scope, identifier, mode)`)
- Apply (`_apply_plan(plan, hard_delete=...)`)
- Result aggregation

The per-noun wrappers fix the scope and translate their identifier into the engine's input. This preserves structural-spec §8.1 ("the cleanup cascade starts from Project") while exposing noun-shaped entry points.

### 9.1 Session usage

Per sub-spec 01 §6, `session` is read at function entry:
```python
def delete(experiment_id: str, *, apply=False, hard_delete=False, session=None) -> CleanupReport:
    s = session if session is not None else automl.session()
    # s.dry_run dictates which universe to clean
    # s.config provides MLflow URI, GCS bucket, project name
    ...
```

No `dry_run` parameter on the function — read from `session.dry_run`.

**Session is required.** All three `delete()` functions require an active session — either passed explicitly or established via `automl.use_project(...)` before the call. Non-session callers (admin scripts, hook subprocesses) must establish a session first. The cleanup engine does not accept raw `bind()`-only state because it needs `session.config.project_name`, `session.config.gcs_bucket`, `session.config.gcs_prefix`, `session.config.mlflow_artifacts_destination` — values that live on `ProjectConfig`, not on `_Bound`.

**Experiment/trial wrappers implicitly scope to the session's project.** Neither `experiment.delete(experiment_id, ...)` nor `trial.cleanup.delete(run_id, ...)` takes a `project_name` argument. They read project from `session.config.project_name` and construct paths from there. If a caller passes an `experiment_id` (or run_id whose parent experiment) belongs to a different project, the lookup will fail or the consistency check (§9.2) will raise `ProjectError` — never a silent wrong-prefix delete.

### 9.2 Trial-delete: parent-experiment lookup mechanism

`trial.cleanup.delete(run_id)` has only the run_id, but the cascade needs to know the run's parent experiment (to compute GCS prefixes and the local sandbox path).

The lookup uses a dedicated seam method — `mlflow.trial.get_parent_experiment(run_id) -> ParentExperimentRef` (added to sub-spec 02 §6.3.3) — which returns the parsed routed-name fields as a typed value. Cleanup never calls the raw MLflow API directly; the parsing lives in the seam.

```python
def delete(run_id: str, *, apply=False, hard_delete=False, session=None) -> CleanupReport:
    s = session if session is not None else automl.session()

    # Resolve the parent experiment via the seam (one call, typed return)
    try:
        parent = automl.mlflow.trial.get_parent_experiment(run_id)
    except StorageError as e:
        raise ProjectError(f"run_id {run_id!r} not found") from e

    # Verify project matches session
    if parent.project_name != s.config.project_name:
        raise ProjectError(
            f"run_id {run_id!r} belongs to project {parent.project_name!r}, "
            f"current session is for {s.config.project_name!r}"
        )

    # Verify mode matches session (the §4.3 consistency rule)
    if parent.dry_run != s.dry_run:
        raise ProjectError(
            f"run_id {run_id!r} not found in current session's mode. "
            f"Did you {'omit' if parent.dry_run else 'mean to use'} --dry-run?"
        )

    # Now we have everything: parent.experiment_id is the AutoML experiment id;
    # parent.mlflow_experiment_id is what `mlflow gc` needs if --hard-delete.
    # Build plan, optionally apply.
    ...
```

**Why a dedicated seam method, not `mlflow.client.raw()`:** sub-spec 02 §5 calls out `raw()` as a signal that the wrapper surface needs to grow. Trial-delete's lookup is a real, recurring need — fold it into the seam so the parsing logic + return type live in one place (the only place that knows MLflow's experiment-name routing convention).

---

## 10. Result shape (SETTLED 2026-05-22)

Three typed dataclasses, each with `schema_version: int = 1` and a `from_dict` classmethod per sub-spec 02 §8 (forward-compatible loader; strips unknown keys).

```python
# project/cleanup.py — types live with the cascade engine

@dataclass(frozen=True)
class CleanupPlan:
    """What WOULD be deleted. Built by every delete() call (with or without --apply)."""
    schema_version: int = 1
    scope: str = ""                          # "project" | "experiment" | "trial"
    identifier: str = ""                     # project name | experiment_id | run_id
    dry_run: bool = False                    # snapshot of session.dry_run at plan-build time

    # MLflow targets — (name, mlflow_experiment_id) pairs, captured at plan-build time so the
    # gc step (if --hard-delete) can use the IDs without re-querying.
    mlflow_experiment_targets: list[tuple[str, str]] = field(default_factory=list)
    # MLflow run targets (used only at trial scope — experiment-scope cascades via delete_experiment).
    mlflow_run_targets: list[str] = field(default_factory=list)

    # GCS prefix patterns to scan + delete (not enumerated blobs — blob enumeration happens at apply time).
    gcs_prefix_patterns: list[str] = field(default_factory=list)   # "gs://bucket/prefix/..."

    local_paths: list[str] = field(default_factory=list)            # absolute local paths to rmtree

    @classmethod
    def from_dict(cls, payload: dict) -> "CleanupPlan": ...

@dataclass(frozen=True)
class CleanupResult:
    """What WAS deleted. Populated only when apply=True."""
    schema_version: int = 1
    mlflow_experiments: dict[str, str] = field(default_factory=dict)   # name → status
    mlflow_runs: dict[str, str] = field(default_factory=dict)          # run_id → status (trial scope)
    mlflow_experiment_ids: list[str] = field(default_factory=list)     # MLflow internal IDs we soft-deleted (for gc)

    gcs: dict[str, int | str] = field(default_factory=dict)            # prefix_pattern → deleted_count | "failed: ..."
    local: dict[str, str] = field(default_factory=dict)                # path → status

    # Hard-delete subprocess result (populated iff hard_delete=True)
    mlflow_hard_delete_status: Literal["skipped", "success", "failed"] | None = None
    mlflow_hard_delete_output: str = ""                                # raw stdout/stderr from `mlflow gc` (or empty)

    @classmethod
    def from_dict(cls, payload: dict) -> "CleanupResult": ...

@dataclass(frozen=True)
class CleanupReport:
    """What every delete() returns. Always has plan; result populated iff applied."""
    schema_version: int = 1
    plan: CleanupPlan = field(default_factory=CleanupPlan)
    applied: bool = False
    result: CleanupResult | None = None

    @classmethod
    def from_dict(cls, payload: dict) -> "CleanupReport": ...
```

**Field naming clarification:**
- `mlflow_experiment_targets` carries `(name, id)` pairs at plan time so the apply step has IDs ready for `mlflow gc` without re-querying.
- `gcs_prefix_patterns` carries the prefix patterns the apply step will scan; **actual blob enumeration happens at delete time, not at plan time**. This is intentional: blob lists can be large and stale (blobs added between plan and apply would be missed). The preview output therefore shows prefix patterns ("will delete all blobs under `gs://bucket/foo/`"), and the exact deleted-count lands in `CleanupResult.gcs` after apply.
- `mlflow_experiment_ids` on the result records which IDs were soft-deleted in this cascade — that's the list passed to `mlflow gc --experiment-ids ...` when `--hard-delete` is set.
- `mlflow_hard_delete_status` is the typed status programmatic callers branch on; `mlflow_hard_delete_output` is the raw subprocess output for human eyes.

CLI `--json` flag serializes `CleanupReport` to stdout using `dataclasses.asdict`. Without `--json`, the CLI prints a human-readable summary.

---

## 11. Orphan handling (SETTLED 2026-05-22)

### 11.1 Orphan cleanup is always mode-scoped — strict isolation never breaks

Cleanup at any scope (project / experiment / trial) only ever touches the `<route>` matching the session's mode. An orphan blob in the *opposite* universe is never reached by a single invocation. To clean orphans in both universes, run cleanup twice — once with `--dry-run`, once without.

This is a direct application of §3.3's strict-isolation principle: dry_run and real are two different containers; cleanup respects the boundary even for orphan handling.

### 11.2 In-universe orphans get picked up by the normal prefix-scan

Within the session's universe, cleanup picks up orphans naturally — no separate "orphan-only" mode required.

How: the GCS delete step lists every blob under `gs://<bucket>/<gcs_prefix>/<route>/...` and deletes them all — MLflow-known or not. Blobs that don't have a corresponding MLflow tag pointing at them (orphans from a partially-failed prior write) get cleaned the same way as MLflow-referenced blobs.

This matches legacy behavior (`_delete_gcs_prefix(bucket, object_prefix)` lists-and-deletes without checking MLflow) and resolves sub-spec 02 §13 item 6:

> *"The cleanup verb (sub-spec 03) handles the 'repair' case by removing orphans first."*

Concretely: if `mlflow.trial.artifacts.write_predictions(...)` writes to GCS but fails before logging the URI to MLflow, the blob is orphaned. A subsequent `automl trial delete <run_id> --apply` in the same session mode removes it as part of normal cleanup — same prefix-scan, same delete loop.

### 11.3 Cross-universe / cross-experiment orphan scanning is out of scope

If a workflow surfaces requiring "scan the whole bucket for blobs older than N days that have no MLflow reference, regardless of mode or experiment," add a dedicated `automl mlflow scan-orphans` admin verb at that time. Cleanup at the project/experiment/trial scope already covers the in-scope-and-in-mode case.

---

## 12. What this sub-spec defers

To implementation-time decisions (not design):

1. **CLI `--json` exact output format.** Pretty-printed vs single-line JSON. Whether to include the plan when `--apply` was passed. Implementation detail of `cli/<noun>.py`.

2. **MLflow `gc` failure recovery.** `--hard-delete` calls `mlflow gc` as a subprocess; if that fails (network, permissions), surfacing the error vs retry is implementation-detail of the cascade engine.

3. **Local-path-not-found vs filesystem error.** `rmtree` failure modes (`PermissionError` vs `FileNotFoundError`) and how each maps to status strings.

To future sub-specs:

4. **Trial sandbox dir path layout** — carry-back to sub-spec 08 (Runner). See §3.4. Cleanup respects whatever path layout the runner picks; the path-segregated proposal is the cleanup-side preference.

To follow-demand additions:

5. **`--only-layers` flag** for layer-selective cleanup. Shape recorded in §4.5; not shipped.
6. **`--confirm-name` guard** for catastrophic-project-delete safety. Shape recorded in §5.2 / §8; not shipped.
7. **`automl mlflow scan-orphans`** admin verb for cross-scope orphan hunting. Mentioned in §11; not shipped.
8. **`automl mlflow gc`** admin verb for cleaning soft-deleted MLflow records left behind by partial-failure `--hard-delete` runs. Mentioned in §6.3; not shipped.
9. **`--all-projects` bulk delete.** Legacy supported this; new design intentionally drops it. Add a sibling `automl project list` + `xargs` loop or a dedicated `--all` flag if real demand emerges.

### 12.5 Legacy symbol disposition

Every public symbol in legacy `automl/cleanup.py` and `automl/trial/cleanup.py` mapped to its new home:

| Legacy symbol | Disposition | New home |
|---|---|---|
| `RouteCleanupTarget` (dataclass) | Folded | Replaced by `CleanupPlan.mlflow_experiment_targets` + `CleanupPlan.gcs_prefix_patterns` in `project/cleanup.py` |
| `RunCleanupTarget` (dataclass) | Folded | Replaced by `CleanupPlan.mlflow_run_targets` + `CleanupPlan.gcs_prefix_patterns` for trial scope |
| `MlflowDeleteResult` (dataclass) | Folded | Replaced by `CleanupResult.mlflow_experiments` + `CleanupResult.mlflow_experiment_ids` |
| `CleanupPlan` (legacy) | Renamed + redesigned | New `CleanupPlan` in `project/cleanup.py` (different schema; see §10) |
| `build_cleanup_plan` (function) | Renamed | Private `_build_plan(scope, identifier, session)` in `project/cleanup.py` |
| `apply_cleanup_plan` (function) | Renamed | Private `_apply_plan(plan, hard_delete, session)` in `project/cleanup.py` |
| `require_confirmation` (function) | Dropped | Per §5.2 — no interactive confirmation; `--apply` is the gate |
| `run`, `_build_parser`, `main` (argparse) | Removed | Per §11.4 of structural spec — no `__main__` blocks in library; CLI dispatch in `cli/<noun>.py` |
| Trial-cleanup's `cleanup(project, project_root, trial_id, run_id, dry_run, confirm_project)` | Renamed + simplified | `trial.cleanup.delete(run_id, *, apply, hard_delete, session)`. Dropped: `trial_id` slug selector (per M3 — run_id is always assignable at trial creation); `dry_run` arg (read from session); `confirm_project` (no interactive confirmation). |

All `[?]` rows in `migration-checklist.md` for `automl/cleanup.py` and `automl/trial/cleanup.py` flip to `[ ]` or `[-]` based on this table.
