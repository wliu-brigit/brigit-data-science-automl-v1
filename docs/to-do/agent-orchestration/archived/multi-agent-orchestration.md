# Multi-Agent Orchestration — Adding Roles Beyond Proposer/Coder

## Status

Priority: P2. Deferred design-revisit. A third agent role (e.g. a **reviewer**
between proposer and coder) is plausibly wanted later, but the orchestration is
currently spread across the library, the CLI, and skill prose in a way that makes
"add a role" a scavenger hunt. This note captures the current wiring and the
options so a future planning session starts with the full picture.

This is a future-work assessment captured on **2026-05-29** (branch
`refactor/four-layer`) during a structural review. It is **not authoritative**
and does **not** prescribe a solution. Re-read the referenced files before
acting; line numbers will drift.

**Relationship to the cleanup pass:** two review findings are *prerequisites* for
this iteration — (a) collapse the duplicated proposer/coder roster into a single
`agent/roles.py`, and (b) split `agent/timeline.py` and drive its phases from the
role set rather than hard-coding two. Doing those in the cleanup leaves the agent
domain in the state this design needs. See "What to do when picked up."

## Problem

The loop is intentionally **LLM-driven, not a state machine** (`CLAUDE.md`:
"the loop is LLM-driven"). `automl experiment run` builds **one** `claude`
subprocess (`agent/launch.py::build_launch`) and exits; that subprocess is the
**manager** (main session), and it drives the turn order by following
`skills/automl/SKILL.md` prose, calling the **subagents** (`proposer`, `coder`)
via the Task tool.

Consequences of that (correct) choice:
- The **sequence** ("proposer → validate → persist → create trial → coder") lives
  only in `SKILL.md` prose (steps 8–11). The library owns *none* of it.
- The **role set** is hard-coded in ~4 library/script sites plus the two
  `agents/*.md` files.
- The **handoff contract** is a single type (`Proposal`) that bakes in
  "proposer writes it, coder reads it."
- `agent/timeline.py` independently hard-codes the two-phase shape in ~15 places.

So adding a role touches many places, and the library "isn't enough" on its own —
exactly the friction that motivated this note.

## How a turn is wired today (verify before trusting)

Two kinds of actor:
- **Manager** = the main Claude session running `skills/automl/SKILL.md`. Not a
  subagent. `models.manager` sets its `--model`/`--effort` (`launch.py:60-63`).
- **Subagents** = `proposer`, `coder`, registered via the `--agents` JSON that
  `build_launch` constructs (`launch.py:122-135`) by parsing
  `agents/automl-{proposer,coder}.md` and overlaying `models.{proposer,coder}`.

Where each concern lives:

| Concern | Location |
|---|---|
| Which roles exist | `ModelsConfig` fields `manager/proposer/coder` (`run_config.py:165-176`, validated against the literal tuple `("manager","proposer","coder")`); `launch._model_settings`+`_agent_overrides` (`launch.py:91-135`); `skills/automl/scripts/render_context.py:116-117` |
| What a role *is* (prompt/tools) | `agents/automl-proposer.md`, `agents/automl-coder.md` |
| Role model routing | `ModelsConfig` + `launch._agent_overrides` |
| Handoff contract | `agent/proposal.py::Proposal` (one type, proposer→coder) |
| **Sequence** (order/handoff/stop) | `skills/automl/SKILL.md` prose, steps 8–11 — **prose only** |
| Timeline reconciliation of the sequence | `agent/timeline.py` — two phases hard-coded in ~15 spots (`:166-169`, `:504-514`, the `("proposer","coder")` loop at `:690`, `:862-863`) |

The library owns *fragments* of "what is a role" and **none** of "what is the
chain." The chain is prose, with the timeline's two-phase shape as a shadow copy.

## Worked example: insert a reviewer (proposer → reviewer → coder)

What you would touch *today*, grouped by difficulty:

**Mechanical but smeared (just to make the role exist):**
1. `ModelsConfig` (`run_config.py:165-176`) — add `reviewer: ModelRoute` + add
   `"reviewer"` to the validation tuple. Every project `config.py` must now
   declare `models.reviewer` (a schema break — acceptable under no-back-compat,
   but real).
2. `agents/automl-reviewer.md` — new role definition.
3. `launch.py` — add `"reviewer"` to `_model_settings` (`:94-97`) **and** the
   `_agent_overrides` mapping (`:124-125`).
4. `render_context.py:116-117` — add reviewer to emitted model config.
5. (optional) `skills/reviewer/SKILL.md` — a manual hatch like propose/coder.

**Genuinely hard (the real work):**
6. **The handoff contract.** `Proposal` is the *only* inter-agent type and is
   proposer→coder-specific. Two shapes:
   - *Gentlest:* reviewer is a **Proposal → Proposal** transform (reads, returns
     a revised proposal). Coder contract unchanged; only add a "re-persist
     revised proposal" step. Reuses `Proposal`.
   - *Heavier:* reviewer returns **approve/reject + comments** → a new
     `agent/review.py::Review` type, a new persisted handoff artifact (the
     `persist_proposal`/`proposal_handoff` path is proposal-specific), maybe a
     `validate review` check, and a **control-flow branch** ("if rejected, loop
     back or stop") — which in the LLM-driven model is more prose.
7. **The timeline.** `agent/timeline.py` hard-codes two phases in ~15 places. A
   third phase touches every one (phase detection `:166-169`, report-path map
   `:504-514`, the publish loop over `("proposer","coder")` `:690`, the backfill
   list `:862-863`). This is the most invasive blocker to *any* new role.

**Prose:**
8. `SKILL.md` steps 8–11 — insert "dispatch `automl-reviewer`; apply/re-persist
   its revision; then dispatch coder." The sequence change is untyped, untested
   prose.

~10 edit sites across config schema, three Python modules, markdown, a contract
decision, and prose. No single home for "the role set" or "the chain."

## Diagnosis

Two concepts are conflated and neither has a home:
- **Role definition** ("what is a proposer") is split across `agents/*.md` +
  `ModelsConfig` field + `launch._agent_overrides` + `proposal.py` + timeline
  phase label.
- **Chain/sequence** ("proposer → coder") lives in `SKILL.md` prose with the
  timeline as a shadow copy.

A clean structure needs one home for each.

## Options (for the future session, not decided)

**A — Role registry; sequence stays LLM-prose.** `agent/roles.py` as the single
source of the role set: typed `AgentRole(name, agent_md, model_route_key,
io_contract, timeline_phase)`. `ModelsConfig` becomes a mapping keyed by
registered roles; `launch`, `render_context`, `timeline` *derive* from the
registry instead of hard-coding `proposer/coder`. Adding a role = one `AgentRole`
entry + an `agents/*.md` + (if approve/reject) a handoff type. Manager still
drives order via prose.
- *Pros:* kills the ~4-site roster duplication (a **today** problem, not
  speculation); "add a role" becomes a one-definition library change; keeps the
  LLM-driven loop intact; minimal new abstraction; makes timeline phases
  registry-driven (needed anyway).
- *Cons:* sequence still prose — a reviewer still needs the `SKILL.md` edit, and
  approve/reject still hand-written branch.

**B — Declarative chain; sequence becomes data.** On top of A, add a typed
`AgentChain` (ordered roles + the handoff contract between each pair) as a
**library default** (`proposer → coder`) that a project *may* override in
`config.py` (`AGENT_CHAIN = ["proposer", "reviewer", "coder"]`). `SKILL.md` reads
the chain and drives it generically; timeline iterates the chain's phases.
- *Pros:* inserting a reviewer becomes a **data + role-definition** change, no
  orchestrator-prose surgery; the chain is one inspectable, testable artifact.
- *Cons:* more abstraction, and a real tension with "loop is LLM-driven." Honest
  middle: **chain = per-iteration role order (data); the LLM still owns iteration
  count, stop conditions, and judgment within each turn** — so it does not become
  the deferred state machine. Requires the generic per-role IO contract (hard
  part #6).

**C — Python driver/state machine.** Sequencing moves into a library `Driver`.
Contradicts the deliberate LLM-driven choice and the `spec/00` §17.12 deferral.
Reject unless that whole decision is being revisited.

## Recommendation captured from the review conversation

- **Do A's `agent/roles.py` registry regardless** — it is dedup of a real ~4-site
  duplication, not speculation, and it makes "add a role" stop being a scavenger
  hunt. It also forces timeline phases to be data-driven (needed anyway).
- **The real work is two things, not the wiring:** (6) generalizing the handoff
  from "the one `Proposal` type" to "each role declares input/output contracts,"
  and (7) de-hardcoding the timeline's two-phase shape. These dominate the effort.
- **Go to B (declarative chain) only if a third role is genuinely near-term.** If
  reviewer is this-quarter, B (with the "chain = per-iteration order, LLM owns
  iteration/stop" framing) is worth designing now against a real second consumer.
  If it is "someday," do A and let B follow real demand
  (`feedback_extension_points_follow_demand`).

## Open decisions to settle when picked up

1. **How near is the reviewer / any third role?** (Decides A vs A+B.)
2. **Should the sequence live in prose (A) or data (B)?** If data: a single
   library-default chain, or project-overridable in `config.py`?
3. **Reviewer IO shape:** Proposal→Proposal transform (gentlest, reuses
   `Proposal`) vs approve/reject `Review` + a loop-back branch?
4. **Generic handoff contract:** do all roles declare typed input/output, with
   the chain wiring output→input? (Needed for B; optional for A.)

## What to do when this is picked up

1. **First land the cleanup prerequisites** (both are independent review
   findings): collapse the proposer/coder roster into `agent/roles.py`; split
   `agent/timeline.py` into `timeline/{ingest,reconcile,publish}.py` and drive its
   phase handling from the role set rather than the `("proposer","coder")`
   literal. After this, the agent domain has a single role-set home and a
   phase-agnostic timeline.
2. **Then** decide A vs B per the open decisions and write a proper planning doc,
   designing the handoff-contract generalization against a real second consumer.
