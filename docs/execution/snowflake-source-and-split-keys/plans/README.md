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
| 1 — keys & naming | [`step-1-keys-and-naming.md`](step-1-keys-and-naming.md) | landed | d43d1ff, dac92c7 | Suite 507-green; `validate project` clean. Deviations: (1) commit 1's `git add -A projects` swept in the untracked fraud scaffold + pre-existing example_homecredit working edits (ModelRoute→opus, notebooks 1–2 churn) — intended consumers/accepted churn, noted for honesty. (2) B9 needed four fixture fixes the plan didn't enumerate (`test_eval_thin_path` + `test_eval_dataset_persistence` `Dataset(...)` gain `split_group_key=`; `test_project_validation` `SnowflakeSource(...)` gains `unique_key=`; augmentations match-string `non-hash-key`→`non-unique-key`). (3) No test pinned the scaffold SQL placeholder (B7 step 2 over-specified) — only `CONFIG_PLACEHOLDERS` updated. (4) Self-review grep clean for code/docs/SQL; stale `hash_key`/`SPLITID` still existed in example_homecredit notebooks at this step boundary (later resolved in tail-end item 2) and in old logged experiment artifacts (old state, untouched per ground rule). Post-landing review session (2026-06-04, 7-angle): NaN holes in both edge validators fixed (84fef01), zero-row materialize now errors; normalizer-unification + duplicate-check consolidation carried into the step-2 header, source-construction validation + collision-guard altitude into the step-3 header. |
| 2 — dataset record & lifecycle | [`step-2-dataset-record-and-lifecycle.md`](step-2-dataset-record-and-lifecycle.md) | landed | fb3d72a, f605b50, f99dd24 | Suite 522-green. Deviations: (1) pre-existing `test_dataframe_content_hash_is_sensitive_to_rows…` pinned row-order sensitivity — its row-order assertion became a changed-rows assertion. (2) Plan's Task 4 round-trip test asserted `read == payload`, contradicting its own implementation (reader injects `record_uri`) — assertion adjusted to expect the injected pointer. (3) `tests/unit/trial/test_manifest.py` pins the runner payload shape the plan didn't list — `data.manifest_uri` → `data.record_uri` in the fixture. (4) `test_one_trial_local.py::…failure_report…` materialized live-GCS at a fixed route with fresh tmp MLflow — re-runs now trip the refuse-to-overwrite guard (record moved to MLflow); switched to the sibling test's unique `dry_run+namespace` route. Its commit-2-era run left two orphan objects in the real bucket at `…/example_homecredit/example-homecredit/data/datasets/v1_da6dbdc5/` (data.parquet, feature_registry.csv; no MLflow record) — **resolved 2026-06-04 on wendao's instruction**: the whole non-dry-run `example_homecredit` GCS route (19 objects: the orphans + historical old-format test state) deleted and verified empty; prod MLflow holds no non-dry-run example_homecredit experiments, so nothing referenced the bytes. (5) Carried-in normalizer unification done: eval `_normalize_unique_key` now delegates to `automl.data.split._normalize_key` (sorted, blank/duplicate-free — composite keys hash identically on both sides). The "consider collapsing ~5 key checks" item was first deferred, then **done same day on wendao's call** (drift risk outweighs the error-contract coupling): one shared implementation in the new `automl/utils/keys.py` (`normalize_key` + `validate_unique_key` with injectable `error_cls` — utils stays a leaf, each edge keeps its exception type: `DataError` at the data edge via `split.validate_unique_key` delegation, `ValueError` in eval). Six copies collapsed (split.py canonical, eval_dataset external+augmentation frames, _load external+augmentation payloads, base.py join — whose missing-key `KeyError` became `ValueError`, unpinned). Eval edges thereby gain the non-null key check; `_normalize_key`'s "module-internal to data.split" pin superseded — the shared home is `utils.keys`, pin test updated. Post-landing review (2026-06-04, 3 agents: lifecycle correctness, rename completeness, plan completeness): no critical/important findings; minor notes carried into the step-3 row. One recommendation was initially rejected because the prod artifact proxy was assumed to 500 on missing paths; later live/file-backed verification resolved it in tail-end item 4, and `list_dataset_records` now raises on real transport failures while returning `[]` for missing `datasets/`. |
| 3 — Snowflake | [`step-3-snowflake.md`](step-3-snowflake.md) | landed | 6102d9d, c1c30c5, 7f654bc | Suite 545-green; e2e test written and verified to *skip* (live run stays tail-end). Deviations (full detail in the plan's "Execution notes" section): (1) **commit re-allocation** — Tasks 7+8 (scaffold, fraud) executed inside commit 2 so the `base_table_sql` field rename landed atomically with every consumer, keeping each commit suite-green; commit 3 = probe/docs/deps/e2e only. (2) `payment_routing` converted too (rename forced it; SQL starters to the SELECT contract, specifics preserved). (3) Task 1 fixture needed `parent.connector` wiring + a `fetch_one` test. (4) Task 4's quoted block didn't exist verbatim — branch placed around the two separated sites; `SPLIT_PCT` added to quality-filter protected cols. (5) Collision-check altitude resolved to the pipeline edge; `add_split_pct`'s dead guard + pinning test removed. (6) E2E env names moved to `AUTOML_SNOWFLAKE_E2E_*` (contracts forbid new `AUTOML_E2E_*`) + `qa` marker; designation via 3 env vars, fresh `DEV_AUTOML_E2E_BASE_<uuid>` per run. (7) Extra sweeps: e2e breadth test (stale kwarg + missing `unique_key`, latent since step 1), mlflow trace-fixture filenames, `.env.example` gains the `SNOWFLAKE_*` block. (8) Heads-up (d) verified — drift hint already matches by prefix; heads-up (e) still blocked by prod proxy 500-on-missing, carried to tail-end. Known-deferred: notebook `1_define_…` still mentions `base_data_sql` (tail-end notebook pass). Post-landing review (2026-06-04, 5 agents: adversarial correctness, cross-source consistency vs CSV/parquet, lifecycle integration, probe/scaffold/docs truth, completeness audit — no critical findings, no silent scope, cross-source parity confirmed structurally clean): fixes landed (a86d6db, suite 554-green) — (a) `_table_exists` now UPPER-folds both comparands (a lowercase `SNOWFLAKE_SCHEMA` in `.env` made the exists-check miss against Snowflake's uppercase-folded INFORMATION_SCHEMA → silent full rebuild on every load); (b) the SELECT-only/collision guards judge a scrubbed statement (string literals masked, inline `--` comments stripped) — inline comments or literals containing `;`/`SPLIT_PCT` no longer false-positive, real emissions still caught; (c) adopt path coerces integral-but-float/Decimal-typed `SPLIT_PCT` to int64 (Arrow fetch dtype risk on the live path); (d) test adds: case-insensitive exists-check, guard scrub cases, SELECT-led multi-statement, construction-edge key validation pinned for all three sources; (e) file sources gain the "path is recipe, content is layer-1" altitude comment. Reviewer suggestion **dropped with reasoning (wendao)**: a starter-comment warning that `base_table.sql` date filters freeze at snapshot-build time — snapshot semantics are core vocabulary, the SQL is legitimate under the contract, and relative windows binding at build time is the designed behavior (step 4's predicates add the dynamic option). Noted, no action: dry-run with a `training_data.sql` omitting SPLIT_PCT errors at the SQL layer (raw "invalid identifier") instead of the friendly post-pull message; plan's "main project dependency" phrasing was loose — `[project].dependencies` is empty, all runtime deps live in `[dependency-groups].dev` by repo convention. |
| 4 — flexible splits | [`step-4-flexible-splits.md`](step-4-flexible-splits.md) | landed | 559723d, 2bf7d9f | Suite 573-green (was 554). Heads-up (a) confirmed — consumers beyond the plan's lists, all swept via `grep -rn "Splits("`/survivor greps: `payment_routing/config.py`, the e2e Snowflake config template, `tests/e2e/test_homecredit_data_model_breadth.py` (ad-hoc `split_range=` slice → predicate), agent fixtures (`test_launch`/`test_timeline`/`test_proposer_context`), `test_one_trial_local.py` loader mocks + eval_loads pin, `test_sources_pipeline_contract.py` contract fixture. Deviations: (1) project configs/scaffold updates (plan Task 5) executed during Task 3 so the runner integration suite could go green mid-step — same commit, order only. (2) `test_run_config.py`'s two overlap-*rejection* tests replaced (overlap validation died with the range API): a record-don't-police pin + a bucket-ranges-now-`TypeError` pin. (3) Eval identity's `_normalize_buckets` (sort + overlap rejection) deleted with no predicate analogue — the AST is hashed verbatim; its test became predicate-required + recipe-based-identity tests; `ev_` ids change (forward-only, per plan). (4) Archive prefix is the convention's full date (`2026-06-04-`, plan guessed `2026-06-`); inbound links in the effort README, design.md, and `out-of-sample-eval…` note retargeted (plan only named HANDOFF/README links; HANDOFF had none). (5) Test-authoring footgun worth knowing: `x == Where("c") < n` is a *chained comparison* (`Where.__eq__` builds a Predicate) — parenthesize the predicate in assertions. Self-review checklist all green: no callables in the split path, replay-from-AST proven by test_trial_replay, `to_pyarrow` compiled but unwired (no `filters=` anywhere), survivor greps empty. Post-landing review (2026-06-04, 5 agents: adversarial predicate correctness, old-way/back-compat hunt, lifecycle integration, design/docs truth, improvements — no critical findings; hard cut confirmed clean in live code; lifecycle chains all held): fixes landed (0aa9a36, suite 585-green) — (a) `EvalDataset.from_dict` no longer silently loads pre-step-4 split_view records (`predicate=None`); loud "re-prepare" error, the one genuinely lenient reader; (b) `Splits.from_dict` bare-mapping fallback removed (canonical `{"predicates": …}` wrapper required) + loud tombstone for old `"ranges"` payloads; (c) comparison predicates reject `None` values at build/rehydrate (`== None` silently matched nothing in pandas — `.is_null()`/`.not_null()` is the vocabulary; deviation from the plan's "values are str/int/float/bool/None" line, which now applies to membership lists only); (d) chained-comparison footgun documented in the module docstring; (e) mask()/to_pyarrow() null-semantics divergence (`!=`, `~(==)` on nullable columns: pandas keeps null rows, pyarrow drops them) documented on `to_pyarrow` as a known gap to reconcile when the push-down reader is wired — mask() is authoritative until then; (f) stale `buckets`/`split_id` vocabulary fixed in `docs/to-do/out-of-sample-eval-and-dataset-management.md`; (g) +12 tests (from_dict rejection paths, nested-not round trip, Splits.from_dict rejections + tombstone, old-record load guard). Deferred to tail-end with reasoning: retired-range contracts ratchet (`RETIRED_EXECUTABLE_PATTERNS`) — its doc scan covers the known-stale example_homecredit notebooks, so it lands with the notebook pass (now tail-end item 5). No-action notes: AST deliberately not an MLflow tag (lives in the contract artifact + eval record; tags stay the small queryable lineage set); empty `isin([])` stays legal (record-don't-police — a deliberately-empty named view is legitimate); mask/to_pyarrow parallel dispatches stay as-is (table-driven refactor judged worse, matches repo taste); no early column-existence validation added (design §12: load is the single edge). **Raised for wendao, not acted on:** design §12/§13/§14 say the record form is "DNF tuples … accepted directly by `read_parquet(filters=…)`" — what shipped (per the reviewed plan) is the nested op/items AST compiling to pyarrow *expressions* (also a valid `filters=` input, strictly more expressive). Plan and code agree; the design wording doesn't. Protocol says design disagreements are raised, not reinterpreted — **resolved same day: wendao approved amending the design**; §12/§14 now describe the nested AST (amendment marked inline, dated). |

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

## Tail-end activities (after step 4) — one batched live session

Tracked here so they aren't lost; not part of any step's green gate.
**Run these as a single session with wendao on the VPN** (wendao
2026-06-04: VPN setup is slow — finish *all* code work first, then do the
live items in one isolated batch rather than scattering them). The session
needs: VPN, `SNOWFLAKE_*` in `.env`, `AUTOML_E2E=1`, and a
wendao-designated tiny `dev_` table (exported as
`AUTOML_SNOWFLAKE_E2E_SOURCE_TABLE` / `_TARGET` / `_UNIQUE_KEY`).

1. **Live Snowflake e2e** — **DONE 2026-06-04** (ran ahead of the batched
   session once wendao configured `.env`; the only connectivity issue was
   the `SNOWFLAKE_ACCOUNT` form — must be the `<orgname>-<accountname>`
   identifier, not the bare account name; documented in `.env.example`).
   Designated source: 1000-row `DEV_AUTOML_E2E_SRC_FCT_LOANS`, sampled
   read-only from `brigit_snowflake.dbt_analytics.fct_loans` (78.8M rows,
   public/read-only role) into the sandbox schema — `LOAN_ID` unique key
   (verified 1000 distinct), `TARGET_DPD45` = `IS_GROSS_DPD45::INT`,
   mature-D45 rows only, 23 positives. `test_materialize_bootstraps_pulls_
   and_attaches` **passed in 91s**: bootstrap (harness DDL + SPLIT_PCT
   injection) → mint v1 → attach-as-pinned → refresh-data dedup back to v1,
   against prod MLflow + GCS. Both `DEV_AUTOML_E2E_*` tables dropped and
   verified gone; the `dev_snowflake_e2e` MLflow/GCS records remain
   (throwaway route; never deleted per ground rule). Side-note for future
   live work: `fetch_df` (Arrow fetch) supports SELECT result sets only —
   use `information_schema` queries instead of `DESCRIBE`/`SHOW`.
2. **Notebook cleanup + execution** — **DONE 2026-06-04** (`de9baa7`,
   `3f65d37`, current pass).
   `example_homecredit` notebooks were updated to current
   `unique_key`/`SPLIT_PCT`/`base_table_sql`/`Where(...)` vocabulary and
   stale cached outputs were stripped. The workflow was renumbered so the
   model-creation alternatives are `3.1_run_agent_automl`,
   `3.2_author_new_trial`, and `3.3_fork_existing_trial`, followed by
   `4_reevaluate_existing_model` and `5_inspect_logged_runs_and_artifacts`.
   `AUTOML_E2E_NOTEBOOKS=1 uv run pytest tests/e2e/test_homecredit_notebooks.py -q`
   passed in 8:48, including the agent-backed `3.1` path.
3. **First real `fraud_anomaly_detection` materialize** against the
   warehouse (fills the TBD placeholders; the duplicate-unique-key
   conversation is expected — that's the check working).
4. **`list_dataset_records` swallow revisit** — **DONE 2026-06-04**
   (`3f65d37`). Verified missing `datasets/` lists cleanly as `[]` against
   file-backed MLflow and the live prod proxy; genuine list transport/auth
   failures now raise `StorageError` instead of being read as empty.
5. **Retired-range ratchet** — **DONE 2026-06-04** (`3f65d37`).
   `tests/contracts/test_skill_commands.py::RETIRED_EXECUTABLE_PATTERNS`
   now rejects the dead split vocabulary (`split_range`, `train_buckets`,
   `test_buckets`, `.buckets(`, `Splits(train=[(`) across skills and active
   project docs/notebooks.
6. Move this effort `execution/ → archive/` per the docs lifecycle.
