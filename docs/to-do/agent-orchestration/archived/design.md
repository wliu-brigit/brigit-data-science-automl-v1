# Agent Orchestration Architecture — Design

> **Superseded by [`architecture.md`](../architecture.md)** — the comprehensive
> current design. This earlier pass uses older vocabulary (AgentRunner, harness,
> runner-owned-by-flow) that later changed (AgentBackend/Compute, runner in the
> library, flow-as-composition, one `run(step)` operator). Kept for history.

## Status & meta

- **Type:** Design doc (architecture + decisions + rationale). **Not** an
  implementation plan — that is a separate follow-up doc.
- **Captured:** 2026-05-29, branch `refactor/four-layer`, from an extended
  design conversation.
- **Supersedes:** `loop-state-machine.md` (P1) and
  `multi-agent-orchestration.md` (P2). Those two notes describe the *same root
  problem* from two angles; this design resolves both. Keep them for history;
  this is the consolidated picture.
- **Authority:** Forward-looking. Re-read the referenced code before acting;
  file paths and line numbers drift. Where this doc names a current file,
  verify it still matches before trusting it.

---

## 1. The problem (why this exists)

Two standing complaints turned out to be one root cause:

- **P1 — `--max-iter` isn't enforced.** The iteration budget and stop
  conditions are *prose the LLM is trusted to honor*, not code. Each trial is
  real compute; "stop at 5" is a cost-control promise the system doesn't keep.
- **P2 — adding a role is a scavenger hunt.** Inserting a reviewer between
  proposer and coder touches ~10 sites (config schema, `launch.py`,
  `render_context.py`, `timeline.py`, the `agents/*.md` files, SKILL.md prose)
  because the *chain* lives only in `skills/automl/SKILL.md` prose and the role
  set is smeared across the library.

**Root cause:** orchestration — turn order, handoffs, stop conditions, context
routing — lives *inside the LLM* (`SKILL.md` prose, executed by a long-lived
"manager" Claude session), instead of in deterministic library code.

Everything else the system feels like — "tangled," "half-CLI-half-skill," "no
visibility," "can't swap Claude Code for Codex" — is downstream of that.

### Requirements scorecard (current vs. goals)

| Requirement | Today |
|---|---|
| Bounded AI / clear scope | ✅ at the *trial* level · ❌ at the *loop* level (prose) |
| Operational guarantees (`--max-iter`, stop-on-fail) | ❌ prose-trusted |
| Visibility of the orchestration | ❌ lives in prose — can't read/test/trace |
| Cheap extensibility (add a role) | ❌ ~10-site scavenger hunt |
| One home for logic / skill = glue | ❌ logic smeared into prose + fragments |
| Durable state = source of truth | ✅ MLflow |
| Runner portability (Claude → Codex) | ❌ Claude Code *is* the orchestrator |
| Dual-mode (headless + interactive) | ⚠️ interactive-only, one mega-session |
| Multiple agent shapes (workflow + ReAct) | ❌ only the proposer→coder workflow |

This scorecard *is* the requirements list. Every design choice below targets a
red cell.

---

## 2. The core principle: orchestration vs. judgment

When work is "given to the agent," two separable things are handed over:

| | What it is | Wants to live in |
|---|---|---|
| **Judgment** | *what* to do — the next hypothesis, the contents of `model.py`, whether to stop | the **LLM** (irreducibly) |
| **Control flow** | *what to do next* — sequencing, stop conditions, retry, ordering, "persist before create" | deterministic **code** (testable, traceable) |

The current system put **both** in the LLM, so control flow inherited the LLM's
properties: no guarantees, no visibility, no tests. The fix is not "less AI" —
it is **pull control flow into the library; leave judgment in the agent.**

Separating them lets us be *more* generous with the agent, not less: once code
owns the rails (the loop, the stop, the contract), the agent can be handed a
larger judgment space inside them, because the rails guarantee it cannot run
away. **Boundaries enable autonomy; they don't trade against it.**

### The recurring move

> Find any "activity" that quietly bundles **judgment + a deterministic
> effect**, and split them.

- The **manager** bundled *orchestrate (control) + judge (judgment)* → split:
  orchestration becomes Python; judgment becomes discrete agent calls.
- The **coder** bundles *write `model.py` (judgment) + run it (effect)* → split:
  the coder only writes; the orchestrator runs (see §5, Decision D2).

Apply this test to every future agent.

### Industry grounding

This is the mainstream direction, not a bespoke invention:

- Anthropic's *Building Effective Agents* distinguishes **workflows** (LLMs
  orchestrated through predefined code paths) from **agents** (LLM directs its
  own process). The AutoML loop is really a **workflow** with judgment at each
  node. Its five patterns map directly: proposer→coder = prompt chaining,
  reviewer = evaluator-optimizer, "pick the right agent" = routing,
  investigation fan-out = orchestrator-workers.
- **LangGraph** formalizes the same as a state graph (nodes = agents/decisions,
  edges = control flow, one checkpointed state object).
- **GSD-2** (a Claude Code workflow system) made exactly this move: GSD-1 was
  LLM-as-controller; GSD-2 pulled flow into a file-driven state machine that
  dispatches fresh agent sessions with context pre-inlined.
- **Durable-execution engines** (Temporal, etc.) split work into *deterministic
  workflow code* and *non-deterministic activities* — the same partition as
  control-flow-vs-judgment.

(See **§12** for how Uber, Meta, Block, and Google independently built this same
shape — and the cheap refinements worth stealing.)

---

## 3. The layered architecture

The package is a **layered dependency architecture**. Each layer depends only
on layers below it. The one invariant that keeps it from re-tangling:

> **Dependencies point DOWN only — never up, never sideways through prose.**
> (Today's tangle *is* a violation: orchestration leaked *up* into the skill
> and *sideways* into prose.)

```
 ┌─ INBOUND ADAPTERS (thin doors — translate only, no logic) ───────────────┐
 │   cli/ verbs   ·   skills/ (shell out to CLI)   ·   cron / tests           │
 └───────────────────────────────┬──────────────────────────────────────────┘
                                  │ calls
 ┌─ L4  ORCHESTRATION — agent/ ───▼──────────────────────────────────────────┐
 │   orchestrator (the loop, --max-iter, checkpoints) · registry(roles) ·     │
 │   activities(proposer/coder/…) · contracts(Proposal | ControlSignal)       │
 └──────────┬────────────────────────────────────────────┬───────────────────┘
            │ calls run_trial()                           │ calls run(role, ctx)
 ┌─ L3 EXEC ▼ runner/ ─────────────┐       ┌─ OUTBOUND PORT ▼ AgentRunner ─────┐
 │   assembles ONE trial run        │       │   → claude -p | codex | api        │
 └──────────┬──────────────────────┘       └────────────────────────────────────┘
            │ uses
 ┌─ L2  LIFECYCLE — project · experiment · trial ────────────────────────────┐
 │     (validate/ cross-cuts · Session threads through as context)            │
 └──────────┬─────────────────────────────────────────────────────────────────┘
            │ uses
 ┌─ L1  LEAF DOMAINS — data · eval · model · utils ──────────────────────────┐
 └──────────┬─────────────────────────────────────────────────────────────────┘
            │ reads / writes durable state
 ┌─ L0  SEAMS — mlflow (durable state)  ·  gcs (heavy bytes) ─────────────────┐
 └────────────────────────────────────────────────────────────────────────────┘
```

Notes:
- **L1–L4** are the four levels: leaf domains → lifecycle → runner → agent.
- **mlflow is the state *seam*, not a peer leaf.** It depends on nothing in the
  package but is read/written by every layer as the durable backbone.
- **`Session`/`project`** is foundational context that threads through all
  layers rather than sitting cleanly at one level.
- **Outbound ports** (the hexagonal "driven" side): `AgentRunner` (→ the LLM
  backend), `mlflow`, `gcs`. These are where the system reaches the outside
  world, and where backend swaps (Claude Code → Codex) are absorbed.

---

## 4. Key components (L4)

| Component | Responsibility | Today → Target |
|---|---|---|
| **Registry** (`agent/roles.py`) | Single home for "what agents exist": `AgentRole(name, prompt, tools, allowed_reads, io_contract, shape)` | New. Replaces the role set smeared across `launch.py`, `ModelsConfig`, `render_context.py`, `timeline.py`. |
| **Activities** (`agent/activities.py`) | One stateless function per role: `run_proposer`, `run_coder`, `run_investigator`. Each = `AgentRunner.run(role, ctx)` + validate-vs-contract. | New (extracted from SKILL.md prose + `proposer_context.py`). |
| **Orchestrator** (`agent/orchestrator.py`) | The loop: ordering, `--max-iter`, stop conditions, checkpoints. Owns control flow. | New. Re-homes the deleted `scripts/loop_state.py` as a package module. |
| **AgentRunner** (`agent/runner_port.py`) | Outbound port: `run(role, ctx) -> output`. Backends: `claude -p`, `codex`, raw API. | New. This is the seam that earns the Codex-swap. Replaces the hardcoded manager-spawn in `agent/launch.py`. |
| **Contracts** (`agent/contracts.py`) | Typed, versioned handoff artifacts. Generalize `Proposal` → `Proposal \| ControlSignal`, and per-role input/output contracts. | Extends existing `agent/proposal.py` (schema_version 2). |
| **Runner** (`runner/trial.py::run_trial`) | Execute ONE trial (data→fit→eval→log). | **Already exists as a library function.** Becomes a code-triggered *activity* the orchestrator calls (Decision D2). |

**Role = portable data, not a skill.** A role is plain definition (prompt +
tools + contract). A *skill* is one *invocation adapter* over a role (its
Claude-Code interactive face); a headless call is another. Making "skill" the
role's identity would re-couple to Claude Code and undo the Codex-swap — so the
role stays backend-agnostic, and skills/headless are adapters onto it.

### The runner is a *harness* — and the agent↔runner coupling is the boundary

The coder's job is clean only because `runner/trial.py` is **opinionated**: it
owns "load training data → fit → eval → log," leaving exactly one hole —
`model.py`. The agent's bounded job *exists only relative to that harness*. So:

> The unit of bounded work is a **task-type = (agent-role + harness + contract)**,
> coupled on purpose. The coupling **is** the boundary — don't decouple agent
> from runner; what's pluggable is the *triple*, composed by the orchestrator.

The runner is therefore not one fixed thing — it's a **family of harnesses**, and
"runner" generalizes to **execution boundary**, on a spectrum of opinionatedness:

```
 none / pure judgment      sandboxed tools           tight pipeline
 ───────────────────       ───────────────           ──────────────
 proposer, selector        investigator, EDA         coder, HPO, feature-eng
 reads context, returns    read-mostly + scratch     fills ONE slot in a fixed
 a typed artifact          exec; NON-prod sink       load→do→eval→log; prod sink
 ◄─ more AUTONOMY ──────────────────────────────────────── more CONTROL ─►
```

Two consequences:
- **Two pluggable seams, not one:** `AgentRunner` swaps the *brain*
  (claude/codex); the **harness** swaps the *hands* (trial-runner / HPO-sweep /
  sandbox / none). External tools (optuna, hyperopt) plug in *as harnesses*
  behind a contract.
- **The output *sink* is a first-class, per-task, safety decision.** Tight
  harnesses write to production MLflow; **generic/investigation agents write to a
  non-production sink** (scratch / findings / just-return) and run read-mostly —
  that sink + capability-scope *is* their boundary, replacing the opinionated
  pipeline.

**The runner refactor is part of this story.** HPO / feature-engineering /
ablation runners are the sibling-harness work in
[`multi-runner-architecture.md`](../../multi-runner-architecture.md); it and this
agent design meet exactly here — every new executing task-type is a
(harness, role, contract) triple.

---

## 5. Control flow: the loop and its checkpoints

```python
# agent/orchestrator.py  (illustrative; names are sketches)
def run_experiment_loop(session, max_iter, checkpoints):
    for i in range(max_iter):                      # ← the --max-iter GUARANTEE (code)
        ctx       = build_context(session)         # ← re-hydrate from MLflow (L0)
        proposal  = run_proposer(ctx)              # activity (judgment)
        if isinstance(proposal, Stop): break
        proposal  = checkpoints["proposal"](proposal, ctx)   # handler: none/human/agent
        trial_dir = create_trial(proposal)         # L2 lifecycle (deterministic)
        outcome   = run_coder(proposal, trial_dir, ctx)      # activity: WRITE model.py only
        checkpoints["code"](trial_dir, ctx)        # handler: none/human/agent
        result    = run_trial(trial_dir, session)  # ★ ORCHESTRATOR runs the runner (D2)
        # runner logs result to MLflow; next turn's build_context() reads it
```

### Checkpoints with pluggable handlers

The orchestrator is **one** function with **checkpoints** at the seams
(after-proposal, after-code). Each checkpoint has a **handler**, and autonomy is
a *parameter*, not a second orchestrator:

```
 proposer ─[ proposal-review ]─ coder ─[ code-review ]─ run_trial ─ result
                  │                          │
            handler =                  handler =
         · none      (autonomous)      · none      (autonomous)
         · human     (interactive)     · human     (interactive)
         · reviewer-agent (FUTURE)     · code-reviewer-agent (FUTURE)
```

This unifies three things into one mechanism:
- **Autonomous run** — handlers auto-pass.
- **Interactive review** ("here's the proposal / the `model.py` — change it
  before I run?") — handler = ask the human.
- **Future reviewer agents** — the human slot and the agent slot are the *same
  checkpoint*, filled differently. Adding a reviewer later plugs a handler into
  a seam that already exists; the orchestrator does not change.

### Pause / resume

"Pause at a checkpoint" = run to the checkpoint, **persist to MLflow**, surface
the artifact, exit. The human edits `model.py` on disk; the next invocation
**resumes** from the checkpoint. Consistent with the durable-state model — no
long-lived blocking process.

**Implication for loop state:** the resumable state must encode the current
*phase/checkpoint*, not just the iteration count — e.g.
`{iteration: 3, phase: "awaiting-code-review", trial_dir: …}`, derived from
MLflow (a thin scratch cache is allowed but not authoritative). This is the
returning `loop_state.py`, slightly richer than the deleted original.

---

## 6. Invocation: three doors, one logic path

All entry points are **thin adapters** over the same core. No logic in any
adapter.

| Door | Mechanism | Who drives the loop |
|---|---|---|
| `automl experiment run` (CLI) | calls `run_experiment_loop` | code — autonomous, guaranteed |
| `/automl run` (interactive skill) | shells to the same CLI; **streams** output into the session | code; human watches / Ctrl-C / reviews at checkpoints |
| `/automl propose`, `/automl investigate` (interactive) | calls a **single** activity | the human sequences manually |

### Dual-mode without forking

- **Autonomous:** Python orchestrates; activities run headless
  (`claude -p`/codex); guarantees enforced; visibility via surfaced artifacts +
  MLflow.
- **Interactive:** the *human* (or a thin skill) drives; activities can stream
  into the session so the human sees the agent think and can Ctrl-C / review at
  checkpoints. Same registry, same contracts; no hard loop-guarantee because the
  human is the guarantee.

**Visibility:** structured outputs (the `Proposal`, the `TrialResult`) and
durable artifacts are always surfaceable. Streaming the headless subprocess
output recovers live "watch it think" for cheap (one flag). A second invocation
adapter (native in-session subagent) for *mid-reasoning interjection* is
deferred until demanded.

---

## 7. Where things live (prompts, skills, packaging)

- **Prompts move into the library** (e.g. `automl/agent/prompts/*.md`). A prompt
  is a role *definition* = library data. The registry indexes them.
- **Skills become thin** — ~3 lines that name CLI verbs. They carry no prompt
  text, no chain order, no contract knowledge.
- **The CLI is the seam.** Skills are markdown and cannot import Python, but they
  *can* shell out to `uv run automl …` (they already do, via
  `render_context.py`'s `safe_commands`). The CLI verb layer is the bridge
  between the Claude-Code-plugin world and the Python-library world.
- **`agents/*.md` at the plugin root** become optional: the headless and
  interactive-via-CLI paths inject prompts from the library. Keep a plugin-root
  agent file only if you also want a *native Task-tool subagent* for ad-hoc use;
  if so, it's a thin mirror kept honest by a contract test.

### Packaging & distribution

One repo, **two distributable artifacts**, one logic core, bridged by the CLI:

```
brigit-automl/
├── .claude-plugin/plugin.json   ← Claude Code PLUGIN root  ┐ plugin
├── skills/   (SKILL.md → uv run automl …)                  │ artifact
├── agents/   (optional native mirrors)                     │
├── hooks/                                                   ┘
├── pyproject.toml                                           ┐ library
└── automl/  (agent/ · cli/ · runner/ · …)                   ┘ artifact (pip/uv)
```

Install = two actions: (1) `uv tool install brigit-automl` → CLI + library on
PATH; (2) `/plugin install brigit-automl` → skills/agents/hooks. The plugin's
skills depend on the CLI from step 1. This is the conventional **"thin client
over a CLI"** shape (editors over linters, `gh` + extensions).

---

## 8. Decision log

| # | Decision | Rationale | Rejected alternative |
|---|---|---|---|
| **D1** | Orchestration lives in **code**, not LLM prose. | Control flow wants guarantees + visibility + tests, which prose can't give. Resolves P1 and P2 at once. | Keep LLM-driven loop (status quo) — fails the cost-control promise and keeps "add a role" a scavenger hunt. |
| **D2** | The **orchestrator** runs the runner; the **coder only writes `model.py`** (never executes). | Responsibility follows the layer — execution belongs to the assembling layer. Tightens the agent's boundary (it can't run code), and the coder's anxious execution prose ("run once, no pipes, stop on nonzero") becomes a code guarantee. The coder is already one-shot today, so almost nothing is given up. | Coder runs the runner (status quo) — wider blast radius, determinism only prose-enforced. **Nuance:** ReAct agents (investigator/debugger) *do* run effects inside their own bounded loop — that's what ReAct is. So "who triggers the effect" is **shape-dependent**: one-shot agents → orchestrator; ReAct agents → the agent, via narrow code-owned tools. |
| **D3** | Role = **portable data**; the LLM backend sits behind an `AgentRunner` outbound port. | Makes Claude Code / Codex / raw API swappable backends. Today Claude Code *is* the orchestrator, so no swap is possible. | "Role is a skill" — re-couples to Claude Code, kills the swap. |
| **D4** | Handoffs are **typed, versioned contracts** (`Proposal \| ControlSignal`); consumers depend on the *contract*, not the producer. | Decouples agents (coder doesn't know who made the proposal). Makes "insert a reviewer" a `Proposal→Proposal` transform with coder unchanged. Validation at the boundary fails fast. | Untyped "read whatever it output" — brittle, silent drift. |
| **D5** | **Checkpoints with pluggable handlers** (none / human / agent). | One orchestrator parameterized by pause-policy gives interactive review *and* future reviewer agents through the *same* seam, without forking the orchestrator. | A second "interactive orchestrator" — two control-flow homes to keep in sync (worse than today's one). |
| **D6** | Prompts in the **library**; skills thin; **CLI is the seam**. | Enforces "skill = glue, not interface" (the existing three-tier rule) by giving logic a home to delegate to. | Logic in skill prose (status quo) — the tangle. |
| **D7** | Chain as **explicit code** first; chain-as-data only on real demand. | Explicit function calls are readable, typed, debuggable. Generalize to a declarative `[PROPOSER, CODER]` chain only when ≥2 real chains justify it. | Build the declarative chain engine up front — abstraction tax before a second consumer. |
| **D8** | Interactive = **stream** first; native-subagent adapter later. | Streaming recovers "watch it think" for one flag. Mid-reasoning interjection (the costly second code path) waits for demand. | Build both invocation mechanisms now — two paths to keep contract-honest before they're needed. |
| **D9** | **MLflow stays the durable state**; loop-state encodes *phase*, derived from MLflow. | Keeps the one ✅ we already have. The orchestrator reads counts/phase from MLflow; local scratch is a cache, never authoritative. | A local JSON source of truth — re-invents state MLflow already holds. |

---

## 9. Integration with the current system (current → target → future)

| Piece | Today | Target change |
|---|---|---|
| `data · eval · model` (L1) | ✅ solid | none |
| `mlflow · gcs` (L0) | ✅ durable state | none — keep as source of truth |
| `project · experiment · trial` (L2) | ✅ solid | none |
| `runner/` (L3, `run_trial`) | ✅ library function | becomes a code-triggered **activity** the orchestrator calls (D2) |
| `agent/` (L4) | ⚠️ `proposal.py`, `launch.py`, `proposer_context.py`, `timeline.py`; **orchestration in SKILL.md prose** | **add** orchestrator · registry · activities · `AgentRunner` · generalized contracts; **prompts move in** |
| `agent/launch.py::build_launch` | spawns one `claude` manager subprocess with `--agents`; invokes `/brigit-automl:automl …` | **replaced** by the code orchestrator + `AgentRunner` (per-activity headless calls) |
| `skills/automl/SKILL.md` | ~200-line prose protocol (steps 8–16) | shrinks to ~3 CLI lines |
| `skills/automl/scripts/` | `render_context.py`, `preflight.py` | thin / largely absorbed into CLI verbs |
| `agents/*.md` | role defs at plugin root | become **library prompts** (optional CC mirror) |
| `--max-iter` | ❌ prose-trusted | **code-enforced** — `loop_state.py` returns as the orchestrator's phase-aware state |
| `hooks/` + `agent/timeline.py` | `SubagentStart/Stop` hooks capture the timeline | timeline becomes **orchestrator-owned telemetry** (it knows when each activity starts/ends); the CC subagent hooks no longer fire in the headless path. Re-home this. |
| `cli/` | thin verbs | + verbs: `propose`, `investigate`, (maybe) `run-trial` |
| `tests/contracts/` | forbids the retired `loop_state.py`; pins the two-phase shape | **un-retire** `loop_state.py`; retire the manager-launch path; drive phase tests from the registry |

**Future (deferred — follow demand, do not build yet):**
- More roles: reviewer, investigator, fixer (registry entry + prompt + contract).
- A **router** activity (pick the right agent for a task) — the routing pattern
  over the registry.
- Chain-as-data (D7), Codex backend (D3), native-subagent interactive adapter
  (D8), a **governed learning store** (agent-proposed, code-gated, versioned,
  falsifiable — never the source of truth).

---

## 10. Key concerns, and where the design answers each

| Concern (raised in design) | Addressed by |
|---|---|
| System feels tangled; cross-cutting changes touch many places | Layering + dependency-direction rule (§3); one home per concept (registry, orchestrator, contracts) |
| "Add a role" is a scavenger hunt | Registry (D-component) + typed contracts (D4) + checkpoints (D5): a reviewer ≈ one orchestrator line + a registry entry + a prompt |
| Keep the AI bounded / in scope | Tighter capabilities (coder can't execute, D2); contracts gate handoffs (D4); rails enable autonomy (§2) |
| Don't over-constrain the agent / maximize potential | Judgment stays fully in the agent (§2); ReAct agents keep their own bounded loops (D2 nuance) |
| Visibility | Orchestration is code (readable/testable/traceable); streaming + MLflow artifacts (§6) |
| Swap Claude Code → Codex | `AgentRunner` outbound port (D3) |
| Multiple agent types / levels / a router | Registry + shapes (workflow vs ReAct) + future router activity (§9 future) |
| One system, few caller points | Ports-and-adapters: one core, thin doors (§6) |
| Don't lose interactivity | Checkpoints + streaming + dual-mode (§5–§6) |
| Where prompts live / how chaining works / install | §7 (library prompts, CLI seam, two artifacts) |
| `--max-iter` / stop control | Code-enforced loop (D1, D9) |
| Who runs the runner | Orchestrator (D2) |

---

## 11. Non-goals, open questions, risks

**Non-goals (for this design):**
- The step-by-step implementation plan (separate doc).
- The governed learning store (deferred; left as a seam, not built).
- A heavyweight orchestration framework (Temporal/LangGraph as infra). Borrow
  the *concepts*; implement at the weight of one process spawning subprocesses.

**Open questions for the implementation doc:**
1. Exact module boundaries inside `agent/` (one `orchestrator.py` vs.
   `orchestrator/` package; where the phase-state lives).
2. `AgentRunner` backend interface shape — what `ctx` carries, how structured
   output is requested/validated per backend.
3. Timeline/telemetry migration: precisely what the orchestrator records now
   that subagent hooks don't fire in the headless path.
4. Contract-test changes: which retired-path entries to relax, which new
   invariants to pin (registry-driven phases).
5. Whether `experiment run` keeps any legacy `claude`-launch path during
   migration, or cuts over cleanly (no-backcompat stance favors a clean cut).

> **Bigger, architecture-level open questions** — concurrency, orchestrator shape
> (loop vs. scheduler vs. graph), a first-class capability layer, observability,
> and event-driven operation — are captured in
> [`open-questions.md`](open-questions.md), surfaced after this design settled.

**Risks:**
- **Schema break:** project config grows a role registry. Acceptable under the
  no-backcompat stance, but every project `config.py` is affected.
- **Cold-start cost:** fresh headless sessions re-ingest context per step.
  Negligible when each trial is minutes of compute, but real.
- **Two invocation mechanisms** (if/when the native-subagent adapter lands) must
  be kept contract-honest — the typed contract + a contract test are the
  guard.
- **Harness co-evolution (the swap isn't free):** models are post-trained against
  particular harnesses and develop dependency on them. The `AgentRunner` swap
  (Claude → Codex) abstracts the *invocation*, but prompts/harness tuned to one
  backend may degrade on another — budget for per-backend prompt adaptation.
- **The agent stops before deployment — on purpose.** Even Meta's REA is scoped
  to *experimentation*, not shipping to production serving. Our deployment
  constraints (single Docker, cloudpickle, SQL-canonical) keep the agent out of
  the deploy/serving step; that boundary is a feature, and a known limit.

---

## 12. Industry validation & refinements to adopt

A 2026 scan of how others build agentic systems shows this design is
**convergent**, and our "execution boundary" has an industry name: the
**harness** ("Agent = Model + Harness"). Where they go further, there are cheap
refinements worth taking.

**Convergence — independent arrivals at the same shape:**
- **Meta KernelEvolve** literally uses a **"job-harness"**: the harness compiles +
  evaluates each candidate and feeds *rich diagnostics* back to the LLM, which
  generates the next candidate. That is our judgment/harness split, named.
- **Meta Ranking Engineer Agent (REA)** runs the ML-experimentation lifecycle for
  ads ranking — hypothesis → train → debug → analyze → iterate (= proposer /
  runner / investigator / loop) — **scoped to one codebase**, with **compute
  budgets confirmed up front and runs halted at threshold** (= code-enforced
  `--max-iter`), on an internal framework ("Confucius") with an SDK to job
  schedulers + experiment tracking (= our library + `AgentRunner` + CLI seam).
- **Uber** orchestrates dev agents with **LangGraph** (orchestration-as-graph =
  our code orchestrator), grounds the LLM in **deterministic tools** (static
  linters feed the graph), and exposes an **encapsulated interface** so the
  security team writes rules "without understanding the AI architecture."
- **Block Goose**: any-LLM provider abstraction (= `AgentRunner`) and multiple
  surfaces — desktop / CLI / API (= our three doors).
- **Google's self-evolving recsys**: an **Offline agent** (sandbox, proxy
  metrics) vs an **Online agent** (production, business metrics) — our
  dry-run/production split.
- The **harness thesis**: ~70% of agent performance lives *outside* the model,
  and harness engineering beats model-tier upgrades (a ~16-point spread on
  identical weights). This is the case for *this whole refactor*.

**Refinements to adopt (cheap, high-value):**
1. **Rich diagnostics, not just a metric.** KernelEvolve feeds structured
   bottleneck analysis back. Our `TrialResult` contract should carry
   diagnostic-rich output (failure class, profile), not only `primary`.
2. **Step/compute budgets on *every* loop**, including ReAct agents — not only
   `--max-iter` on the modeling loop. Generic agents get a step budget.
3. **Risk-tiered checkpoint handlers.** REA auto-handles routine failures within
   guardrails and escalates the rest; the harness literature uses plan-before-act
   + a Safe/High-Risk reviewer. Our `none/human/agent` handler (D5) should be
   chosen **by risk tier** — auto-gate low-risk, escalate high-risk.
4. **A plan-approval checkpoint up front.** REA has engineers approve the
   exploration plan *before* autonomy starts — a natural top-of-loop checkpoint.
5. **Ground in deterministic tools** wherever one can verify what the LLM would
   otherwise infer (Uber's linters-into-the-graph).
6. **Make the registry/contract a clean extension surface** a data scientist can
   use *without* understanding the orchestrator (Uber's encapsulated interface).
7. **Reconsider the governed learning store's priority.** REA's "curated
   historical insights database" is exactly the store we deferred (§9) — a real,
   shipped consumer suggests it earns its place sooner than "someday."

---

## 13. Glossary

- **Activity** — a single agent call (or trial run); the non-deterministic unit
  the orchestrator sequences.
- **Orchestrator** — deterministic library code that owns the loop, stops, and
  checkpoints.
- **AgentRunner** — the outbound port that runs a role on a backend
  (`claude -p` / codex / api).
- **Role** — a portable agent definition (prompt + tools + allowed-reads +
  io-contract + shape), held in the registry.
- **Contract** — a typed, versioned handoff artifact (`Proposal`,
  `ControlSignal`, `TrialResult`); the seam that decouples agents.
- **Checkpoint** — a seam in the loop with a pluggable handler
  (none / human / agent).
- **Shape** — how known an agent's path is: *workflow* (known steps, e.g.
  proposer→coder) vs *ReAct* (open-ended, e.g. drift investigation).
- **Harness** — the execution boundary a role runs inside, on a spectrum: *none*
  (pure judgment) → *sandboxed tools* (read-mostly + scratch, non-prod sink) →
  *tight pipeline* (opinionated load→do→eval→log, prod sink). The runner is the
  tight end; the "hands," swappable independently of the "brain" (`AgentRunner`).
- **Task-type** — a coupled (agent-role + harness + contract) triple; the unit of
  bounded work the orchestrator composes.
- **Sink** — where an activity's output goes (production MLflow / scratch /
  findings / just-return); a per-task safety choice, central to bounding generic
  agents.
