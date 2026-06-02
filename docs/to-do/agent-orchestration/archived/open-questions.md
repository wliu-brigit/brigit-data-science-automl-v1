# Agent Orchestration — Open Architectural Questions

> **Folded into [`architecture.md`](../architecture.md) §14** (with several now
> resolved — search-strategy, capability=tools, linear↔parallel, compute). Kept
> for history.

Surfaced during the design conversation (2026-05-30), **after** the core design in
[`design.md`](design.md) settled. These are **not decided** — they're
architecture-level questions to iterate on, kept separate so `design.md` stays the
decided source of truth.

For each: the question, why it matters, our current lean, defer-or-now, and — most
importantly — the **cheap "door to keep open"** so a future decision isn't a
rewrite.

---

## Q1 — Concurrency: parallel exploration

**Question.** The orchestrator is a sequential `for` loop — one trial at a time.
Should it run multiple trials/proposals **in parallel** to speed the discovery
phase?

**Why it matters.** Training is the slow part; running several candidates
concurrently could compress discovery a lot. Every production system we looked at
explores in parallel (Meta REA's parallel validation phase, KernelEvolve's
parallel tree search, Uber's 100× parallel execution).

**Lean / status.** Worth doing eventually; not v1. The risk is that a *blocking*
`for` loop is hard to parallelize after the fact — concurrency touches the
orchestrator core, the harness execution model, and the loop-state model.

**Door to keep open (cheap insurance).** Describe the orchestrator as an
**activity scheduler** (dispatch activities, collect results) that *happens to run
one linear path in v1* — not as a hardcoded `for` loop. Then parallel dispatch is
a change of *strategy*, not a rewrite of the engine.

## Q2 — Orchestrator shape: loop vs. scheduler vs. graph

**Question.** Should control flow become an explicit **graph** (à la LangGraph),
or stay a linear sequence?

**Lean / status.** **Skeptical of adopting a graph framework now.** A graph earns
its complexity *only* when concurrency or branching is real — so this folds into
Q1. For a mostly-linear `propose → code → run` pipeline, a graph is overhead.

**Door to keep open.** Same as Q1 — keep the orchestrator a general scheduler.
Don't adopt a graph engine until Q1 forces branching/parallel topologies; reassess
LangGraph vs. a thin home-grown scheduler at that point.

## Q3 — Search strategy as a separate layer? (resolved: NO)

**Question.** Should "how to explore the experiment space" be a first-class layer
(tree search / bandit), separate from the proposer?

**Resolution (concession from review).** Mostly **no**, for two reasons:
1. The proposer is **already an agentic loop**, not a greedy one-shot — it reads
   context and reasons toward the next move. It *is* an agentic search.
2. The model space is narrow (LightGBM / XGBoost), so **rich domain reasoning
   beats algorithmic search** — there isn't much exotic search to do.

The only residue worth keeping is **parallel/budget allocation across several
agentic proposals**, which folds into Q1 — not a separate search algorithm.
Recorded so we don't re-litigate it.

## Q4 — A first-class capability / tool layer

**Question.** How are agent tools/capabilities **defined, registered, scoped, and
sandboxed** — instead of an ad-hoc per-role list?

**Why it matters.** For tight-pipeline agents the harness *is* the capability, but
for read/ReAct agents the **toolset is the boundary**. Concretely:
- the proposer should be **read-only** (e.g. may query MLflow, may **not** write);
- capabilities must be **explicitly assigned per role**, so scope is legible;
- the same layer is **reused across agents** (proposer, incident/investigation,
  future roles) rather than re-built each time.

**Lean / status.** **Real work to do** — agreed. Lean toward an MCP-shaped,
per-role, explicitly-scoped capability registry (aligns with Claude Code / Codex /
Goose). This is the layer the generic/conversational future leans on hardest.

**Door to keep open.** Make "tools" a **scoped, per-role capability set** in the
registry from the start (even if v1 only grants "read MLflow" to the proposer) —
not a hardcoded list buried inside each agent.

## Q5 — Monitoring / observability of the system

**Question.** How do we get **visibility into what the agents did and what
happened** — beyond per-trial MLflow logging?

**Why it matters.** There's no system-level view of agent runs, decisions,
failures, or cost yet. An autonomous loop you can't observe is hard to debug or
trust.

**Lean / status.** A gap to work on. The orchestrator-owned timeline
(`design.md` §9) is a start; the broader observability surface is open.

**Door to keep open.** Have the orchestrator emit **structured telemetry** per
activity/checkpoint from day one — it's the natural collection point.

## Q6 — Always-on / event-driven operation

**Question.** Should the system run as an **always-on, event-driven service**
(scheduled, drift-triggered) instead of invoke-and-exit?

**Why it matters.** Monitoring/drift/continuous-improvement and the conversational
vision both eventually want triggers, not manual invocation.

**Lean / status.** A good **extension point**, lower priority — add when a real
trigger use case lands.

**Door to keep open.** The scheduler + `AgentRunner` seams shouldn't assume "a
human typed a command" — a cron or an event should be just another inbound
adapter.

## Q7 — Security / trust layer (deployment-dependent)

**Question.** Do we need a guardrail layer — prompt-injection defense, an agent
being manipulated by data it reads, PII, agent identity (à la Uber's AI Gateway)?

**Lean / status.** Low risk for a single-tenant dev tool; real once the agent
reads production/customer data or becomes conversational. Deferrable, but named so
it isn't a surprise. It attaches at the `AgentRunner` / capability-layer seams
(Q4).

---

## Priority read

- **Act-now insurance** (cheap, prevents one-way doors): keep the orchestrator a
  **scheduler**, not a `for` loop (Q1/Q2); make **capabilities per-role scoped**
  in the registry (Q4); emit **structured telemetry** (Q5).
- **Build when demanded:** parallel dispatch (Q1), graph topologies (Q2),
  event-driven triggers (Q6), security layer (Q7).
- **Resolved:** no separate algorithmic search layer (Q3).
