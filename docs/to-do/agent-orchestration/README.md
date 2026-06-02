# Agent Orchestration — refactor dossier

**Front door.** This folder holds the design for reshaping how brigit-automl runs
its agent loop. **[`architecture.md`](architecture.md) is the single active doc —
start there.** Everything else (earlier passes + the original notes) lives in
[`archived/`](archived/), kept for history and **not maintained** — don't read it
unless you have a specific reason.

This README is intentionally **thin and decision-free**: it states the *problem*
and *what we're looking for* — nothing about *how* we solve it. The "how" (the
architecture and the decisions) lives in [`architecture.md`](architecture.md),
the source of truth and may change over time. Keeping decisions out of the front
door means this page never goes stale or contradicts the design.

## The problem (high level)

Today the agent loop is driven by an LLM following prose in
`skills/automl/SKILL.md`: a long-lived "manager" session decides the turn order,
the handoffs, and when to stop. So the **orchestration** — control flow that
wants to be deterministic — lives *inside the model*. Consequences:

- Stop conditions (`--max-iter`, stop-on-failure) are *trusted*, not *enforced*.
- Adding or reordering a role (e.g. a reviewer) means editing many scattered
  places.
- The flow is hard to see, test, or reproduce — and is tied to one agent runner.

## What we're looking for (goals)

Not solutions — the bar any solution should clear:

- Stop / iteration limits are **enforced**, not hoped for.
- Adding an agent or changing the chain is **cheap and local**.
- The orchestration is **visible** — readable, testable, traceable.
- The AI stays **bounded** to its job, while keeping full room for judgment.
- The agent backend is **swappable** (Claude Code today, Codex/others later).
- It works both **autonomously** and **interactively**.
- One home for logic; thin entry points.
- A **pluggable compute layer** — heavy training runs local or in the cloud.
- Each agent is **bounded by its tools** (its capability scope).
- **Concurrency** — many trials at once to speed discovery, statelessly.

## What's here

| Path | Role |
|---|---|
| [`architecture.md`](architecture.md) | **The one active doc — source of truth.** The comprehensive settled architecture: engine/agents/backends/flows, compute layer, tools/capability, concurrency, the decision log (§13), the **must-design-before-build** list (§14), and industry validation. |
| [`archived/`](archived/) | **Historical, not maintained.** `design.md` (earlier pass), `open-questions.md` (folded into §14), and the two original notes — `loop-state-machine.md` (P1, "`--max-iter` isn't enforced") and `multi-agent-orchestration.md` (P2, "adding a role is a scavenger hunt"). Don't trust these over `architecture.md`. |

Everything now lives under this one folder on purpose: once the refactor lands,
all of it belongs here as one record rather than scattered across `docs/to-do/`.

## If you're starting execution

Read [`architecture.md`](architecture.md) end to end — its decision log (§13),
repo layout (§12), execution model (§7), and especially **§14 "must be designed
before the build"** are where you start. `archived/` is context, not
instructions. The implementation plan is a **separate** doc, not yet written.
Verify all file/line references against current code before trusting them.
