# AutoML Refactor

Rebuild of the **brigit-automl** library under a new four-layer architecture with six
canonical noun domains. Phase 7 cutover removed the frozen `automl_legacy/` reference tree;
the fresh `automl/` package is now the only package in this worktree. No old↔new bridges;
clean cut.

**This folder is the single home for the refactor.** Two subfolders:
- **`spec/`** — the **design**: *what* we're building and *why*. Frozen reference.
- **`plan/`** — the **execution**: *how*, in *what order*, *current status*, *what to do next*. Living.

---

## ▶ Status (2026-05-28)

| Stage | State |
|---|---|
| Design — all sub-specs `00`–`11` | ✅ **approved** |
| Final cross-doc carry-back + consistency pass | ✅ **complete** (all open items resolved; `spec/open-questions.md` is **CLOSED**) |
| Implementation strategy (approach, dependency graph, phased roadmap (0–7)) | ✅ **signed off** (`plan/implementation-strategy.md`) |
| Phase 0 — completeness audit + acceptance-checklist | ✅ **done** (symbol coverage complete; behavior gates authored) |
| Phase 0 — scaffold (worktree + freeze legacy + skeleton) | ✅ **done** (commit `281ef80`, branch `refactor/four-layer`, worktree `../automl_dev-refactor`; `import automl` green) |
| **Phase 1** — walking skeleton (one real Home Credit trial) | ✅ **done** (A1.1-A1.4 passed; external Home Credit gate green on 2026-05-27) |
| **Phase 2** — data/model breadth | ✅ **done** (A2.1-A2.5 passed; external Home Credit gate green on 2026-05-27) |
| **Phase 3** — eval breadth | ✅ **done** (A3.1-A3.2 passed; external eval gate green on 2026-05-28) |
| **Phase 4** — experiment/trial reads + cleanup | ✅ **done** (A4.1-A4.4 passed; external QA cleanup gate green on 2026-05-28) |
| **Phase 5** — agent domain + hooks | ✅ **done** (A5.1-A5.3 passed; external agent/hook gate green on 2026-05-28) |
| **Phase 6** — surface breadth + isolation | ✅ **done** (A6.1-A6.4 passed; external namespace/dry-run gate green on 2026-05-28) |
| **Phase 7** — cutover | ✅ **done** (A7.1-A7.4 passed; external final loop + preservation gates green on 2026-05-28; legacy trees deleted) |
| Final whole-refactor audit | ✅ **done** (spec/docs, migration, architecture/safety, full local, and Phase 7/6/5/4/3 external gates green on 2026-05-28) |
| Follow-up spec coverage/readiness review | ⚠️ **open items recorded** (accepted gates still green; branch is committed but not merge-ready until `plan/final-review-open-items.md` is triaged) |

**Where the build lives:** branch `refactor/four-layer` in worktree
`/Users/zhengisamazing/1.python_dir/brigit/automl_dev-refactor/` — the fresh `automl/` is the
four-layer package; `automl_legacy/` and `tests_legacy/` have been deleted.

**→ Next action:** triage `plan/final-review-open-items.md` and
`plan/spec-coverage-review.md` before any merge/cutover decision. The branch is committed and
the accepted gates were green at final audit, but it is **not merge-ready yet** because the
follow-up review found deferred/spec-gap items that need an explicit owner decision.

For future harness gates, use the original MLflow setup on `http://127.0.0.1:54321` and load the
local `.env` in this worktree, copied from `/Users/zhengisamazing/1.python_dir/brigit/automl_dev/.env`.
The detailed Phase 1 evidence is recorded in `plan/acceptance-checklist.md`; do not use the
temporary Phase 1 verification server as the future harness default.

---

## Resuming in a new session — start here

Open a fresh session and paste:

> *"Continue the AutoML refactor. Read `docs/superpowers/automl-refactor/README.md` first, then tell me the current status and the next action before doing anything."*

That's the whole bootstrap — this file is the single entry point. Then:

1. **Read this file top-to-bottom** — the **Status** table tells you what's done and the
   **NEXT ACTION**; the **Document map** tells you where everything lives; **Conventions**
   (below) tells you the rules to work under.
2. **To act on the build:** open `plan/README.md` → `plan/implementation-strategy.md` (the
   overarching plan + dependency graph + phase roadmap) and `plan/acceptance-checklist.md`
   (live gate status). The current phase's detailed steps live in `plan/phases/` (written
   just-in-time — if the next phase has no file yet, that's the cue to write it).
3. **For design detail on a domain:** open the relevant `spec/NN-*.md` (index in `spec/README.md`).
   The design is the **agreed starting point** — build to it, don't relitigate settled calls on a
   whim. But it's **not infallible** (it predates any running code): if the build surfaces a real
   gap or error, **stop and flag it** — update the spec deliberately (with the user), don't
   silently deviate *and* don't blindly follow a spec you've found to be wrong.
4. **Before touching code, know:** the migration approach + conventions below (especially:
   ignore the workspace-root `CLAUDE.md` during this migration; `uv`-only; clean-cut, no
   back-compat). Then do whatever the Status table marks **NEXT ACTION** — confirm it with the
  user if it's a repo-mutating or irreversible step.

**What you can rely on:** status here is kept current; the two checklists in `plan/` are the
source of truth for progress (symbols + behaviors). If a doc ever disagrees with this front
door, this front door + `plan/` win for *status*; `spec/` wins for *design intent*.

---

## Working norms (how to work on this — every session)

- **Autonomous phase loop from Phase 5 onward.** Investigate, write the detailed phase plan,
  self-review the plan/design, implement with TDD, do a post-implementation code review pass,
  verify, update docs, commit, then continue to the next phase unless blocked. Do not pause
  between plan and implementation for a manual review unless there is a true blocker or a
  decision that materially changes scope/risk.
- **Resolve ordinary ambiguity with documented decisions.** If something is confusing,
  ambiguous, contradictory, or looks wrong in the design or plan, call it out in the phase plan
  or review notes and make a reasonable documented decision unless it truly blocks progress.
  Stop and ask only for blockers, policy-like decisions, or choices that materially change
  scope/risk.
- **Confirm before irreversible / outward steps.** Repo mutations (the scaffold, `git mv`,
  cutover) and anything hard to undo: confirm with the user first.
- **Keep the docs consistent.** When you change something, update it *everywhere it's
  reflected* in the same pass — the **Status** table here, the relevant **checklist** rows
  (`plan/`), and any cross-referencing doc. Don't leave a status claimed in one place and
  contradicted in another. (This is exactly the kind of drift the final-pass audit caught.)
- **The design is revisable under evidence.** Build to the spec, but if implementation proves
  it wrong/incomplete, **stop, flag it, and update the spec deliberately** (with sign-off) —
  the phases especially are provisional (see "plan at a glance" below).
- **Evidence before "done."** Mark an acceptance row `[x]` only after the harness gate actually
  passes (run it; don't claim). Flip migration-checklist rows only when the code truly lands.
- **Close each phase deliberately.** After a phase gate passes, update this front door,
  `plan/README.md`, `plan/acceptance-checklist.md`, `plan/migration-checklist.md`, the phase
  plan, and `plan/implementation-strategy.md` if the phase boundaries changed. Update `spec/`
  only when running code proves a design contract wrong or incomplete. Then run a stale-status
  scan before handing off to a fresh session.
- **One universe per command** during the build (the `dry_run` / `namespace` isolation rule);
  never conflate modes or namespaces.

---

## Document map

### `plan/` — execution (read these to know what to *do*)

| Doc | Purpose | When to read |
|---|---|---|
| `README.md` | Execution-folder orientation: plan-doc map, status, how a phase runs. | Landing in `plan/`. |
| `implementation-strategy.md` | The overarching plan: migration approach, cross-domain **dependency graph**, the **phased roadmap (0–7)** (vertical-slice-first), acceptance gates, test strategy. Signed off. | First, for the build. |
| `acceptance-checklist.md` | **Behavior** gates (A0–A7) — "can the new tree do X against the Home Credit harness?" Living status. | To see what's done / next gate. |
| `migration-checklist.md` | **Symbol** coverage ledger — every legacy public symbol → ported (`[x]`) or dropped (`[-]`). Cutover blocked while any row is un-dispositioned. | During execution, per domain. |
| `phases/` | Per-phase detailed TDD plans, written **just-in-time** right before each phase runs. | When starting a phase. |

### `spec/` — design (read these to know *what & why*)

| Doc | Purpose |
|---|---|
| `README.md` | Design index — the four-layer architecture, the six nouns, per-sub-spec "what's done," user preferences, workflow. |
| `00-structural-design.md` | Parent design: vocabulary, layers, domains, folder shape, cross-cutting rules. **Source of truth for structure.** |
| `01`–`11-*.md` | Per-domain interface designs (project, mlflow seam, cleanup, validate, data, model, eval, runner, experiment, trial, agent). Frozen once approved. |
| `open-questions.md` | **CLOSED** design-decision record — the rolling tracker of ambiguities, now all resolved (see its "FINAL-PASS CLOSEOUT"). Kept as history; not an open to-do list. |

---

## The plan at a glance (detail in `plan/implementation-strategy.md`)

> ⚠️ **The phases are provisional, not a contract.** The design (`spec/`) is internally
> consistent and Phases 1 through 7 have now been validated against running code. Future work
> should start from the final audit/open-items docs rather than extending the migration roadmap.

**Approach:** side-by-side clean rebuild on a worktree off the current repo; port-and-reshape
into a fresh `automl/` package; atomic cutover completed in Phase 7.

**Build order (dependency-driven):** `utils`/`errors` → `mlflow` seam + `validate` → `project`
→ `data`/`model`/`eval` → `runner` → `trial`/`experiment`/`agent` → `cli`/`hooks`.

**Strategy: vertical slice first, then thicken** — don't build each layer fully bottom-up
(that's "migrating blind"); build the thinnest end-to-end path first, prove integration, then
add breadth.

| Phase | What | Gate (Home Credit harness) |
|---|---|---|
| **0** | Pre-flight: audit + checklists + scaffold | `import automl` on the empty skeleton |
| **1** | Walking skeleton (thin slice through every layer) | **one real trial runs end-to-end** ★ |
| **2** | Data & model breadth (sources, registry, validators, required-transformer gate, profile) | WOE-gated trial + profile |
| **3** | Eval breadth (all metrics, external eval + augmentation, predictions) | external-eval trial |
| **4** | Experiment & trial domains + cleanup cascade | leaderboard/compare; `experiment delete` |
| **5** | Agent domain + hooks | proposer→coder loop emits a validated Proposal |
| **6** | CLI surface + `--dry-run`/`--namespace` isolation | namespaced sandbox runs + cleans |
| **7** | Cutover | checklists green; full e2e; `automl_legacy/` deleted |

---

## Conventions (from the design — apply during the build)

- **Tooling:** `uv` only, project-local `.venv` (`uv add`/`uv run`); no global pip.
- **No back-compat** for persisted state (old MLflow runs / tags / paths); clean cut.
- **`dry_run` is a session container** (top-level `--dry-run`), strict universe isolation;
  `--namespace` is a parallel full-universe isolation prefix.
- **Project-local files mirror core structure** (`projects/<name>/<topic>/…`).
- Detailed preferences + interview style are in `spec/README.md`.

> **Note for a fresh session:** ignore the workspace-root `CLAUDE.md`'s "work in `automl_dev/`"
> pointer while working this branch. The cutover worktree is
> `/Users/zhengisamazing/1.python_dir/brigit/automl_dev-refactor/`, and this README + `plan/`
> are the authority for the refactor.
