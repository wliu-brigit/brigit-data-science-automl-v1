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
| 1 — keys & naming | [`step-1-keys-and-naming.md`](step-1-keys-and-naming.md) | landed | d43d1ff, dac92c7 | Suite 507-green; `validate project` clean. Deviations: (1) commit 1's `git add -A projects` swept in the untracked fraud scaffold + pre-existing example_homecredit working edits (ModelRoute→opus, notebooks 1–2 churn) — intended consumers/accepted churn, noted for honesty. (2) B9 needed four fixture fixes the plan didn't enumerate (`test_eval_thin_path` + `test_eval_dataset_persistence` `Dataset(...)` gain `split_group_key=`; `test_project_validation` `SnowflakeSource(...)` gains `unique_key=`; augmentations match-string `non-hash-key`→`non-unique-key`). (3) No test pinned the scaffold SQL placeholder (B7 step 2 over-specified) — only `CONFIG_PLACEHOLDERS` updated. (4) Self-review grep clean for code/docs/SQL; stale `hash_key`/`SPLITID` remain in example_homecredit notebooks (tail-end pass below) and in old logged experiment artifacts (old state, untouched per ground rule). |
| 2 — dataset record & lifecycle | [`step-2-dataset-record-and-lifecycle.md`](step-2-dataset-record-and-lifecycle.md) | not started | — | — |
| 3 — Snowflake | [`step-3-snowflake.md`](step-3-snowflake.md) | not started | — | — |
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
