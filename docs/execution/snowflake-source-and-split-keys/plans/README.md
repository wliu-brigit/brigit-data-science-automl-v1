# Implementation plans — status & session protocol

The four step plans for this effort, executed **strictly in order, one step
per session**, each ending with the full suite green. This file is the
execution ledger: a fresh session starts here, and every session updates it
before ending.

[`../design.md`](../design.md) stays the source of truth for *what and why*;
each plan is the executable *how* for one step. If they ever disagree, the
design wins — fix the plan, don't reinterpret the design.

## Status

| Step | Plan | Status | Landed commits | Notes / deviations |
|---|---|---|---|---|
| 1 — keys & naming | [`step-1-keys-and-naming.md`](step-1-keys-and-naming.md) | landed | d43d1ff, dac92c7 | Suite 507-green; `validate project` clean. Deviations: (1) commit 1's `git add -A projects` swept in the untracked fraud scaffold + pre-existing example_homecredit working edits (ModelRoute→opus, notebooks 1–2 churn) — intended consumers/accepted churn, noted for honesty. (2) B9 needed four fixture fixes the plan didn't enumerate (`test_eval_thin_path` + `test_eval_dataset_persistence` `Dataset(...)` gain `split_group_key=`; `test_project_validation` `SnowflakeSource(...)` gains `unique_key=`; augmentations match-string `non-hash-key`→`non-unique-key`). (3) No test pinned the scaffold SQL placeholder (B7 step 2 over-specified) — only `CONFIG_PLACEHOLDERS` updated. (4) Self-review grep clean for code/docs/SQL; stale `hash_key`/`SPLITID` remain in example_homecredit notebooks (tail-end pass below) and in old logged experiment artifacts (old state, untouched per ground rule). Post-landing review session (2026-06-04, 7-angle): NaN holes in both edge validators fixed (84fef01), zero-row materialize now errors; normalizer-unification + duplicate-check consolidation carried into the step-2 header, source-construction validation + collision-guard altitude into the step-3 header. |
| 2 — dataset record & lifecycle | [`step-2-dataset-record-and-lifecycle.md`](step-2-dataset-record-and-lifecycle.md) | landed | fb3d72a, f605b50, f99dd24 | Suite 522-green. Deviations: (1) pre-existing `test_dataframe_content_hash_is_sensitive_to_rows…` pinned row-order sensitivity — its row-order assertion became a changed-rows assertion. (2) Plan's Task 4 round-trip test asserted `read == payload`, contradicting its own implementation (reader injects `record_uri`) — assertion adjusted to expect the injected pointer. (3) `tests/unit/trial/test_manifest.py` pins the runner payload shape the plan didn't list — `data.manifest_uri` → `data.record_uri` in the fixture. (4) `test_one_trial_local.py::…failure_report…` materialized live-GCS at a fixed route with fresh tmp MLflow — re-runs now trip the refuse-to-overwrite guard (record moved to MLflow); switched to the sibling test's unique `dry_run+namespace` route. Its commit-2-era run left two orphan objects in the real bucket at `…/example_homecredit/example-homecredit/data/datasets/v1_da6dbdc5/` (data.parquet, feature_registry.csv; no MLflow record) — **resolved 2026-06-04 on wendao's instruction**: the whole non-dry-run `example_homecredit` GCS route (19 objects: the orphans + historical old-format test state) deleted and verified empty; prod MLflow holds no non-dry-run example_homecredit experiments, so nothing referenced the bytes. (5) Carried-in normalizer unification done: eval `_normalize_unique_key` now delegates to `automl.data.split._normalize_key` (sorted, blank/duplicate-free — composite keys hash identically on both sides). The "consider collapsing ~5 key checks" item was first deferred, then **done same day on wendao's call** (drift risk outweighs the error-contract coupling): one shared implementation in the new `automl/utils/keys.py` (`normalize_key` + `validate_unique_key` with injectable `error_cls` — utils stays a leaf, each edge keeps its exception type: `DataError` at the data edge via `split.validate_unique_key` delegation, `ValueError` in eval). Six copies collapsed (split.py canonical, eval_dataset external+augmentation frames, _load external+augmentation payloads, base.py join — whose missing-key `KeyError` became `ValueError`, unpinned). Eval edges thereby gain the non-null key check; `_normalize_key`'s "module-internal to data.split" pin superseded — the shared home is `utils.keys`, pin test updated. Post-landing review (2026-06-04, 3 agents: lifecycle correctness, rename completeness, plan completeness): no critical/important findings; minor notes carried into the step-3 row. One recommendation rejected with reasoning: `list_dataset_records` keeps swallowing list failures → `[]` because the prod artifact proxy 500s on missing paths — a fresh experiment's empty `datasets/` folder is indistinguishable from a transport error there, and the mint path fails safe at the refuse-to-overwrite guard. |
| 3 — Snowflake | [`step-3-snowflake.md`](step-3-snowflake.md) | not started | — | Heads-up from step-2 close (2026-06-04): (a) `load()` already carries `refresh_source` and the `recipe_identity` hook already exists on `DataSource` — step-2 groundwork, the plan's "extend signature" steps are body work only; (b) match plan file references by content, not line number (minor drift: fake source `artifact_files` ~239, `base_data_sql` in data-pipeline.md at 12/68/113-114); (c) `data-pipeline.md` still says `base_data_sql` on purpose — the `base_table_sql` rename is this step's work; (d) align the drift-warning hint string in `pipeline.py:_attach_active` ("base_table.sql changed") with the actual Snowflake recipe key when `recipe_identity` lands; (e) carried diagnosability note: revisit whether `list_dataset_records`' swallow-to-`[]` can distinguish missing-folder from transport failure once the Snowflake hot path lands (blocked today by the prod proxy's 500-on-missing). |
| 4 — flexible splits | [`step-4-flexible-splits.md`](step-4-flexible-splits.md) | not started | — | — |

Status values: `not started` → `in progress` → `landed` (suite green,
commits pushed, HANDOFF updated). Plans written 2026-06-04 from the approved
design + a full code read; independently reviewed by four review agents on
2026-06-04 (findings folded back into the plans).

**Before step 2 lands:** wendao wipes old MLflow/GCS state manually — the
implementation never deletes or migrates old state itself (design §14 ground
rule).

## Protocol for a fresh session

1. **Orient:** read [`docs/HANDOFF.md`](../../../HANDOFF.md), then this
   status table. Your step is the first row that isn't `landed`. If the
   previous row has Notes/deviations, read them — they may invalidate
   assumptions your plan makes about the tree.
2. **Load the context:** read [`../design.md`](../design.md) end to end
   (don't relitigate settled decisions or §15's rejected ideas), then your
   step's plan end to end. Mark this row `in progress`.
3. **Execute** with the superpowers executing-plans skill (or
   subagent-driven-development) — task by task, in order, ticking the
   plan's `- [ ]` checkboxes as you go. The plans are TDD-shaped: failing
   test → implement → green → commit at the marked boundaries.
4. **When reality disagrees with the plan** (a line moved, a helper is
   named differently, a test the plan didn't list breaks): verify against
   the code, **edit the plan file** to match reality, and record one line in
   the Notes/deviations column. The plan is the record of what was actually
   done. If the disagreement is with the *design* (not the plan), stop and
   raise it with wendao — don't work around it.
5. **Ground rules every session inherits** (from the design + wendao):
   - Never delete or migrate existing MLflow runs, GCS objects, or
     warehouse tables.
   - Everything through `uv`; credentials only via `.env` — never handle
     values.
   - `projects/fraud_anomaly_detection/` only where a plan explicitly lists
     it; throwaway experiments use a gitignored `dev_`-prefixed project.
   - Settled calls from the 2026-06-04 conversation are in each plan's
     header — don't reopen them in passing.
   - **Stage commits explicitly — never `git add -A <dir>`.** The `git add`
     lines in the plans are illustrative, not commands to run verbatim: a
     blanket add sweeps pre-existing working-tree edits and untracked files
     into a step commit (this happened in step 1 — d43d1ff carries unrelated
     example_homecredit edits). Commit exactly the files your step changed;
     anything else in the tree is not yours to commit.
   - More generally: plan steps that touch shared state (git, MLflow, GCS,
     warehouse) get a judgment check before running, not just a deviation
     note after. When a plan instruction is unreasonable on contact with
     reality, stop and fix the plan or ask — executing it and logging the
     deviation is not a substitute.
6. **Close out:** full suite green
   (`uv run pytest tests/unit tests/contracts tests/integration`), all plan
   checkboxes ticked, commits made per the plan. Update this table (status →
   `landed`, commit SHAs, deviations) and `docs/HANDOFF.md` (current state +
   the next step as the next action). If you stop mid-step, leave the row
   `in progress` with a note saying exactly where (last ticked checkbox).

## Tail-end activities (after step 4)

Tracked here so they aren't lost; not part of any step's green gate:

- Live notebook verification on `example_homecredit` (notebooks 1–2 churn
  was accepted in review with this as the check). Known breakage to fix in
  that pass: `notebooks/2_run_agent_automl.ipynb` declares
  `Splits(train=[(0, 80)]...)` — invalid after step 4's hard cut.
- First real `fraud_anomaly_detection` materialize against the warehouse
  (fills the TBD placeholders; the duplicate-unique-key conversation is
  expected — that's the check working).
- Move this effort `execution/ → archive/` per the docs lifecycle.
