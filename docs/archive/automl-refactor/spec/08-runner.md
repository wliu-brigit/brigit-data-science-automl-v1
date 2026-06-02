# Sub-spec 08 — Runner Domain

**Status:** APPROVED 2026-05-25 (Q1–Q6 + three-agent review + fixes applied)
**Started:** 2026-05-24
**Topic:** the straight-line trial chain — `data → fit → eval → log`.
**Parent:** `00-structural-design.md`. Consumes carry-backs from 03, 04, 05, 06, 07.

---

## Scope decision (settled first, frames everything)

**Tier 1.** Apply every carry-back *in place*, plus mechanical hygiene; **no**
stage protocol, **no** shared state carrier, **no** pluggable/swappable runner.

The monolith (`_execute.py`, ~1,500 lines) is an intentional straight-line
orchestrator with one global failure boundary (the ~250-line `finally` reads
~20 partial-state locals to assemble a failure manifest + error log and still
close the MLflow run). That coupling is *intrinsic* to "capture whatever ran so
far on failure," not accidental. Decomposing into composable stages would
require a `TrialRunState` carrier + stage protocol + reproducing the global
failure capture — a high-cost, test-heavy effort whose main payoff (a swappable
runner) the user has explicitly disavowed needing. Per
`feedback_extension_points_follow_demand`, that abstraction is **not built**.

**Tier 1 = what we do:**
- Apply carry-backs 04 / 05 / 06 / 07 / 03 (see reconciliation table below).
- Delete dead code carried in `_stages.py`'s star-import.
- Replace the `from automl.runner._stages import *` star-import with explicit imports.
- Rename files to the new domain convention.
- Split `_stages.py`'s grab-bag into cohesive modules. (This is file hygiene —
  carving an oversized helper file — NOT the stage decoupling we're deferring.)

**Deferred north-star (Tier 2):** a stage-pipeline runner with a shared
`TrialRunState` carrier. Re-open at the **tripwire**: a real second runner
shape is needed (e.g. a distributed / remote / batch executor), at which point
the swappability payoff becomes concrete. Recorded here so it is not lost.

---

## Carry-back reconciliation (grounding)

| Source | Requirement | Current code | Net for 08 |
|---|---|---|---|
| **04** phase order | sample-load → prefit → open run → full-load → fit → eval → log; runner builds the 200-row sample and passes `df`/`registry` to `validate.model` | prefit runs first (validate samples internally via `sample_from=`), then mlflow, then full `data_load` | reorder; runner owns sample-load |
| **05** data contract | build `TrialDataContract` from `LoadedSlice` objects; write via `mlflow.trial.artifacts.write_trial_data_contract`; splits resolved at load (`load_dataset` / `load_dataset_by_trial`) | builds `RunDataContract` from `snapshot.split_view` dict; writes via `mlflow.write_data_contract`; splits pre-realized in snapshot | rebuild `data_load` around `LoadedSlice` |
| **06** required-transformer gate | enforced *inside* `validate.model` | not present | no runner change (lives in validate) |
| **07** eval | `EvalSpec.validate_columns` pre-fit (Caller 2); eval stage calls `evaluate()` → `EvalResult`; pass `_model` / `_model_feature_registry` | no pre-fit eval-column gate; `prepare_eval_split_view` + `_evaluate(..., eval_snapshot_id=)`; ad-hoc result object | add pre-fit gate; rename; `EvalResult` |
| **03** sandbox dirs | `projects/<project>/experiments/<route>/<trial_name>/`, route = `[dry_run/]<project>/<id>`; segregated by mode | flat `experiments/<slug>`, created by `trial/creation.py`, not mode-segregated; runner only validates containment (`_execute.py:159`) | **boundary question — open** |

---

## Design questions

### Q1 — trial folder path + creation ownership — SETTLED

**Decision:** adopt sub-spec 03's mode-segregated trial folder path; ownership is
unchanged from today (creation builds, runner executes).

- **Path:** `projects/<project>/experiments/[<namespace>/][dry_run/]<project>/<experiment_id>/<slug>/`
  (the optional `<namespace>/` segment — from `session.namespace` / `--namespace`, sub-spec 01 — makes the local sandbox dir part of the same full-universe isolation as MLflow + GCS; `""` = today's route)
  (was flat `experiments/<slug>`). Mirrors the MLflow/GCS route prefix so the
  real and dry_run universes never collide on disk. Note this adds **two changes**
  vs. today: the `[dry_run/]` mode segment *and* the `<project>/<experiment_id>/`
  nesting depth (today's layout has neither).
- **Creation builds it** — `trial.create` (09; today `trial/creation.py`)
  makes the folder and writes `model.py` / `metadata.json` / `proposal/` /
  `run.py`. The runner is **not** an entry point for folder creation (it must read
  files the creator already wrote). 03's "the runner creates the segregated dirs"
  wording is **corrected** → "the runner owns + enforces the path; creation builds it."
- **Runner verifies it** — today's weak "is `trial_dir` under `experiments/`?"
  check (`_execute.py:159`) is upgraded to verify the folder sits under the
  **mode-correct** route subtree → a real universe-isolation guard (a dry_run
  session refuses a real-route folder and vice versa), consistent with the
  dry_run-container invariant.
- **Single source of truth:** a small path helper lives in **`runner/`**
  (next to the `run.py` template the runner owns and the guard that enforces it).
  `trial.create` imports it via the sanctioned **trial → runner**
  edge to build the folder; the runner uses it to verify.
  *(Was "experiment → runner" before the sub-spec 09 decomposition promoted Trial to a peer.)* `project/` owns the
  *route string*; the runner composes route + `project_dir` + `<slug>` into the path.

**Carry-back:** correct sub-spec 03 §3.4 + open-questions entry wording
("runner creates dirs" → "runner owns/enforces path; creation builds").

### Q2 — trial numbering / identity — SETTLED

**Decision:** the MLflow number query moves to the seam; *when/where* it is
called is unchanged (runner, at exec time).

- **(a) Relocate the query.** `_next_trial_number_from_mlflow` is a pure MLflow
  read (routed experiment → search trial runs → `max(automl.trial.number)+1`). It moves
  into the seam as a typed read `mlflow.experiment.next_trial_number(...)`
  (absorbing `_run_trial_number`; reusing the seam's paginated run-search). Both
  the runner and `trial.create` import it from there — breaking the
  backward `runner → trial` import under the new `trial → runner`
  direction (post-decomposition; was `experiment → runner`). **Carry-back to 02:** add this read to `mlflow/experiment/`
  (02 never accounted for it). Record the symbol's new home in the checklist.
- **(b) Keep exec-time assignment** (status quo). Trial numbers track *execution*
  order — dense, gap-free, none burned on an un-run draft; `metadata.trial_number`
  stays forbidden. The loop is sequential, so `max+1` doesn't race. Folder leaf
  stays `<slug>`; `trial_id` stays `<number>_<slug>`. Creation-time assignment was
  rejected (breaks ordering; no carry-back asks for it).

### Q3 — reordered phase chain (carry-back 04) — SETTLED

**Decision:** adopt 04's locked ordering, re-expressed in 05's split-at-load
vocabulary; phase-1 obtains its real sample via `.head(200)` on a normal slice
load (no new loader API).

```
phase 1  loaded_fit = load_dataset(split_name=run_config.train_split)
         df_pre_fit = loaded_fit.df.head(200); registry = loaded_fit.registry   ← NO MLflow run
phase 2  validate.model(cls, df=df_pre_fit, registry=registry)                  ← pre-fit gate
         (06's required-transformer gate runs INSIDE validate.model)
         fail → _finish_without_mlflow_run (no MLflow record), as today
phase 3  open MLflow run
phase 4  full load (per-slice) → fit → eval → log                              ← MLflow captures failures
```

Settled by carry-backs (explore-not-ask): the ordering + observability split
(04); fit/eval slice **name-pointers** `run_config.train_split` / `eval_split`
(05); gate-inside-validate (06).

- **Slice designation:** the runner follows the `train_split` / `eval_split`
  *name-pointers* into the free-form `Splits` dict — it never hardcodes the
  literal names `"train"`/`"test"`. Defaults are `"train"`/`"test"` so existing
  projects + the Home Credit harness are unaffected; CV-fold projects override
  the pointers. A pointer naming an absent split raises in `Splits.resolve` at
  phase-1 (pre-run) → clean pre-flight failure, **no extra runner guard**
  (no-redundant-guards: the load already gates it).
- **Sample mechanism (fork resolved → A):** `load_dataset(split_name=train_split).df.head(200)`,
  no `nrows`/`limit` loader option. The fit-slice is read once here and again in
  phase 4 (a single-slice double-read 04 already anticipated). A loader `limit=`
  is **deferred** to a measured cost (follow-demand) — recorded as an open item.
- **dry_run:** phase-1 uses `session.dry_run` (was hardcoded `dry_run=True`) —
  04's noted alignment with the dry_run-container invariant. (The old hardcode
  lives in `validate/targets.py`'s sample probe, which 04 deletes; 08 realizes the
  fix by having the runner load the sample with `session.dry_run`.)
- **SIGALRM timeout armed before phase 1.** Today the alarm is set before all work
  (`_execute.py:257`). The reordered chain must keep it **before phase 1** so a
  hanging source at the pre-fit sample load is still interrupted — not deferred to
  MLflow setup (now phase 3).

### Q4 — data_load phase + `TrialDataContract` (carry-back 05) — SETTLED

> **Refined by Q5 → fit-slice-only.** The construction below originally described
> loading fit + eval slices; **Q5 supersedes this**: the runner loads only the fit
> slice, `evaluate()` owns the eval slice, and `TrialDataContract.slices` records
> the fit slice only. Read Q5 for the authoritative shape.

**Construction (A).** Under split-at-load the runner loads **only the slices it
uses** — the fit slice (`load_dataset(split_name=train_split)`) as a `LoadedSlice`
(per Q5, eval is `evaluate()`'s concern) — and **does not load the full dataset**
(today loads `df_data` whole). Full-dataset facts
(`n_rows`/`n_columns`) come from the `Dataset` manifest via `DatasetRef`. Leakage-
safety + efficiency win, exactly 05's split-at-load intent. Then:

- `DatasetRef` ← `loaded_fit.dataset`;
- one `SliceContract(name=slice.split_name, ranges=slice.split_ranges,
  n_rows=slice.n_rows, content_hash=dataframe_content_hash(slice.df))` per loaded
  slice (runner computes `content_hash` via public `utils.hashing` — `LoadedSlice`
  doesn't carry it);
- `TrialDataContract(trial, dataset, splits=run_config.splits.ranges, slices=(fit,))`;
- write via `mlflow.trial.artifacts.write_trial_data_contract(run_id, payload=contract)`;
- set per-slice tags (`data.dataset_id` / `data.identity_hash` / `data.manifest_uri`
  / `data.slice.<name>.{content_hash,n_rows}`).

**`TrialRef` identity conflict — RESOLVED (A), carry-back to 05.** 05's
`TrialRef` collapsed `trial_id`+`run_id` as "duplicates" — they are **not**:
`trial_id = "<number>_<slug>"` (human, ordered, = run_name + `trial.id` tag +
folder leaf) and `run_id` = MLflow UUID are distinct strings, both load-bearing
today (`_execute.py:584-588`), and Q2 keeps `<number>_<slug>`. **`TrialRef` keeps
both fields.** Carry-back to 05 Q9: restore `run_id`. **Why 05 erred:** it read
00 §5's *vocabulary* statement ("a trial run and an MLflow run are the same thing")
as a claim that the `trial_id` *string* equals the `run_id` *string*, and dropped
`run_id` as a duplicate. The concept is right (one trial = one MLflow run); the
identifiers are two distinct strings serving distinct roles (ordered human handle +
run_name vs. the MLflow API primary key for artifact fetch / tag mutation / URL).

### Q5 — eval phase + pre-fit eval-column gate (carry-back 07) — SETTLED

**Boundary: runner = fit-only; eval owns eval-data.** `evaluate(*, session,
model_run_id, eval_dataset_id, _model, _model_feature_registry) -> EvalResult`
**loads the eval dataset itself** (delegating realization to
`data.load_dataset_by_id` per 07 Q1); `_model`/`_model_feature_registry` only skip
the *model* download. The runner never explicitly loads the eval slice. This
resolves three things at once:

1. **Pre-fit eval-column gate runs against the already-loaded fit frame.** Fit and
   eval slices are row-slices of the *same* materialized `Dataset` → identical
   columns, so `EvalSpec.validate_columns(fit_frame, target)` answers exactly what
   the pre-fit gate asks, with no extra eval-slice load. **Clarifies 07's "loaded
   eval frame"** → any loaded slice of the run's dataset (in practice the fit frame
   already in hand); schema-equivalent, strictly cheaper.
2. **Refines Q4 → fit-only slices.** The runner loads **only the fit slice**;
   `TrialDataContract.slices` records the **fit slice only** (the data the model
   trained on). Eval-data lineage lives in `EvalResult.eval_dataset_id` (eval owns
   eval-data integrity via content-addressing + L2, 07 Q2), not the trial contract.
   Per-slice tags = fit slice only. (05's `slices` stays plural for future
   multi-slice fits e.g. CV folds; the straight-line chain populates one.)
   **Reviewed + accepted as intentional** (vs. today's contract recording both
   train+test hashes): nothing is lost — the contract's `splits` field still records
   *all* split ranges (incl. eval), so the eval slice is reproducible from
   `dataset_id` + ranges, and its realized integrity lives in the eval domain
   (`EvalResult.eval_dataset_id` + tags + manifest). Crash-before-eval still has the
   eval *definition* in `splits`.
3. **No eval-slice double-load** — loaded once, inside `evaluate()`.

Settled-by-07 mechanics (explore): `prepare_eval_split_view → prepare_eval_dataset`;
`eval_snapshot_id → eval_dataset_id`; `ctx → session`; returns `EvalResult`;
`mlflow_url` dropped from the type (CLI derives via seam helper).

**Eval call shape:** the runner calls `prepare_eval_dataset(...)` (recipe →
`eval_dataset_id`; for split_view this is recipe-only, **no eager GCS write** —
07 Q2 moved empty-bucket detection publish→first-load), then
`evaluate(session=, model_run_id=, eval_dataset_id=, label="test",
set_as_primary_label=True, _model=model, _model_feature_registry=…) -> EvalResult`.
`evaluate()` loads the eval slice on first use (delegating to
`data.load_dataset_by_id`). (Note: even today the runner doesn't hold the eval
DataFrame — `prepare_eval_split_view` only stages; `evaluate()` loads. The change
is the rename + the no-eager-write delegation, not who loads.)

**Pre-fit gate scope — split_view vs external.** Checking the *fit* frame is
correct **only when the eval dataset is `split_view`** (a slice of the same
`Dataset` → identical columns), which is the Home Credit harness case. For an
**`external`** eval dataset (separately-imported, possibly different columns), the
fit frame can't witness the eval columns; that case is column-checked **inside
`evaluate()`** (07: `EvalSpec.validate_columns` is also used internally) — i.e.
post-fit rather than pre-fit. Named limitation: the runner's pre-fit gate guards
split_view eval; external-eval column drift surfaces at `evaluate()`-time.

**Validation fixture uses a fit-frame sample.** Today the pyfunc round-trip fixture
is built from `df_test_features` + `y_pred` (`_execute.py:895-909`). Under fit-only
loading `df_test` no longer exists, so the round-trip builds its fixture from a
**fit-frame sample** (`head(10)` of the fit slice) — the round-trip only needs
representative real rows with the full column set to verify the persisted pyfunc
model reloads + predicts identically + latency. The `predict` phase predicts on
that fit-frame sample for the expected-scores fixture.

**Train-eval diagnostic — KEPT.** The runner evaluates the eval slice (primary)
*and* the train slice best-effort (failure ignored) for the overfitting signal —
a real, used diagnostic; carry-forward, no carry-back asks to drop it.

**06 (model) — no runner change.** The required-transformer gate is enforced
inside `validate.model` (pre-fit, on the sample fit); the runner's post-fit
contract validation (`_validate_post_fit_registry`, method-contract, required
attrs, single-target) is unchanged.

### Q6 — Tier-1 file decomposition — SETTLED

**No `runner/stages/` folder for Tier 1.** The inconsistency in 00 is **isolated to
the appendix migration table (line ~840)**, which lists `runner/stages/` as a
Tier-1 migration target *without* the "only when a 2nd runner appears" qualifier.
§8.5 and §13.3/§17.2–3 already agree the `stages/` folder + `stages/base.py` `Stage`
ABC are the **deferred Tier-2 shape** (composable stage primitives across multiple
runners) — not built now. Tier 1 splits `trial.py` by ordinary module **cohesion**,
not "stages." **Carry-back to 00:** correct migration line ~840 only (drop
`runner/stages/` from the Tier-1 target; it remains the §13.3/§17 deferred shape).

**Tier-1 layout:**

```
runner/
├── __init__.py        run_trial, TrialResult
├── trial.py           the straight-line chain (was _execute.py)
├── paths.py           trial-folder path helper + universe-isolation verify (Q1)
├── contract.py        post-fit checks (_validate_post_fit_registry, method-contract, attrs, single-target)
├── validation.py      deployment round-trip: fit-frame fixture build + subprocess launch + GCS publish
├── _pyfunc_check.py   the round-trip subprocess (was the 286-line inline string) — extracted (fork A); kept import-minimal (mlflow/pandas/numpy only, NO automl imports — preserves fresh-process isolation)
├── manifest.py        manifest + artifact-listing + failure-manifest assembly; owns the per-slice data-section rebuild (DatasetRef/SliceContract, replacing snapshot_identity_hash/split_view_hash/train_hash/test_hash) + `_run_gcs_prefixes` + BOTH failure paths (main `except` and the `end_run`-failure manifest rewrite); EvalIndex replaces the hand-built `evaluation` dict
├── _modules.py        collision-safe trial-model import (materialize + load) AND code-bundle staging (`stage_code_bundle` for pyfunc `code_paths`, used in the fit phase + best-effort on the failure path) — NOT in validation.py (staging precedes the round-trip)
├── session_lock.py    moved from automl/session/ (00 line 841); CLI `automl trial lock` → runner.trial_lock (00 line 456)
└── template.py        run.py shim (unchanged)
```

Small helpers (`_hash_seed`, `_resolve_git_commit`, `_safe_error_tag`, timing
recorder) stay in `trial.py`.

**Mechanical hygiene (confirmed):**
- Delete dead code (exploration-confirmed unused): `_write_error_log` shim,
  `_load_trial_model_module`, `_frame_shape`.
- Replace `from automl.runner._stages import *` with explicit imports.
- `_execute.py` → `trial.py`; `_stages.py` dissolves into the cohesive modules.
- Move the session lock + rename its CLI surface (00 lines 456/841).
- **(A)** Extract the inline pyfunc round-trip subprocess to `runner/_pyfunc_check.py`,
  launched as a subprocess — readable + testable, same fresh-process isolation.

---

## Consolidated carry-backs (apply at closeout)

| Target | Change |
|---|---|
| **03** §3.4 + open-questions | "the runner creates the segregated dirs" → "the runner owns + enforces the path; `trial.create` builds the folder." (Q1) |
| **05** Q9 `TrialRef` | Restore `run_id` (MLflow UUID) alongside `trial_id` (`<number>_<slug>`) — the "drop as duplicate" audit-trim was an error; they are distinct + both load-bearing. (Q4) |
| **02** `mlflow/experiment/` | Add the typed read `next_trial_number(...)` (absorbs `_run_trial_number`); 02 never homed it. (Q2) |
| **00** migration line 840 | Drop `runner/stages/` from the Tier-1 runner target (flat cohesive modules instead); keep `runner/stages/` noted as the §13.3/§17 deferred Tier-2 shape. (Q6) |

## Net change summary (what 08 does to the runner)

- **Folder path** mode-segregated (Q1); runner *verifies* it (universe-isolation guard).
- **Number query** moves to the mlflow seam; assignment stays exec-time in the runner (Q2).
- **Phase order** reworked to 04's sample→prefit→open-run→full-load→fit→eval→log, in 05's `load_dataset` vocabulary; pre-fit sample = `load_dataset(split_name=train_split).df.head(200)` (Q3).
- **data_load** loads the fit slice only; builds `TrialDataContract` (fit-slice `SliceContract`) via `mlflow.trial.artifacts.write_trial_data_contract`; new per-slice tags (Q4, Q5).
- **eval** via `evaluate(...) -> EvalResult` (eval owns eval-data loading); pre-fit `EvalSpec.validate_columns` on the fit frame; train-eval diagnostic kept (Q5).
- **Decomposition** into `runner/trial.py` + cohesive modules; dead code + star-import removed; session lock moved; subprocess extracted (Q6).
- **No** stage abstraction / pluggable runner (deferred Tier-2 north-star).

---

## Three-agent review — findings applied (2026-05-25)

Independent fresh-eyes (`code-reviewer`) + codebase-comparison (`code-explorer`) +
coverage-validation (`general-purpose`), run in parallel before locking.

**Validated (no change):** the four carry-backs are all correct and necessary; the
`TrialRef` `trial_id`≠`run_id` distinction was confirmed against `_execute.py:584-588`
(fresh-eyes flagged it at conf 95 as *supporting* our carry-back); coverage agent
confirmed all six Q-decisions captured with no invented content.

**One decision taken (user):** eval-slice lineage stays **fit-only** in the trial
data contract — accepted as intentional (see Q5; `splits` ranges + eval-domain
integrity cover it).

**Fixes applied this pass:**
- Validation fixture rebuilt from a **fit-frame sample** (df_test no longer loaded) — Q5.
- Code-bundle staging filed under `_modules.py`, not `validation.py` (staging precedes the round-trip) — Q6.
- `prepare_eval_dataset` call shape clarified (recipe id → `evaluate()`; no eager GCS write for split_view) — Q5.
- Pre-fit gate scope: split_view (fit frame) vs external (checked inside `evaluate()`, post-fit) — named limitation, Q5.
- SIGALRM armed before phase 1 — Q3.
- Q4→Q5 forward-pointer added; path nesting-depth change noted (Q1); 00 inconsistency pinned to the appendix table (Q6); manifest per-slice data-section + `_run_gcs_prefixes` home + both failure paths (Q6); `_pyfunc_check.py` import-minimal note (Q6); 05 carry-back rationale strengthened (Q4); dry_run-hardcode location noted (Q3).

## Carry-backs applied in the final cross-doc pass (2026-05-27)

- **`AUTOML_DRY_RUN` → `AUTOML_INHERIT_DRY_RUN` (sub-spec 11 #5).** The env var the
  launcher sets and the runner + hook read is **transport-only** (it ferries the
  parent's `session.dry_run` across `subprocess.run`, which can't carry a contextvar) —
  renamed to stop reading as a system-wide *mode*. The runner decodes it into its own
  `session.dry_run`; nothing else routes on it.
- **Delete the metadata-conflict check (`runner/_execute.py:289`).** Legacy code raised
  when the env `dry_run` disagreed with the trial-metadata `run_mode`/`dry_run`. Sub-spec
  10 §7.1 dropped `run_mode`/`dry_run` from `TrialMetadata` (universe = path + session),
  so there is nothing left to conflict with — the check is removed (`feedback_no_redundant_guards`).
  The path-based universe-isolation guard (Q1) is the remaining mode check.

## Open items (carried to implementation / other sub-specs)

- 🔵 **`RunDataContract.to_split_view()` consumers — RESOLVED (→ 05).** Enumerated in the
  final pass: the method is **dropped wholesale, not "ported."** Its callers were never
  the runner — they were the data-internal contract validators (`validate_run_data_contract`
  → L1 `validate_trial_data_contract`; `validate_split_view` → L3 `verify_loaded_slice`)
  and the inspect/replay path (`load_data_snapshot` → `load_dataset_by_trial` + per-slice
  field access). Content-addressed `SliceContract.content_hash` + L1–L4 replace the legacy
  `split_view`-dict + `view_hash` reconstruction, so no `to_split_view()` equivalent is
  needed on `TrialDataContract`. The runner builds the contract from `LoadedSlice` directly
  (Q4). Recorded in 05's "Dropped fields" list.
- ⚪ **Loader `limit=`/`nrows` for the pre-fit sample** — deferred until the fit-slice
  double-read is a *measured* cost (Q3; follow-demand).
- ⚪ **Test-suite sweep for dead-code deletions** — `_write_error_log` shim /
  `_load_trial_model_module` / `_frame_shape` removal must check `tests/` for
  indirect callers (impl-time).
