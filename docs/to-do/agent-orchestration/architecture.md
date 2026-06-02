# Agent Orchestration — Architecture (comprehensive)

## Status & meta

- **Type:** The settled architecture. A **superset** of everything in this
  dossier — it supersedes [`design.md`](archived/design.md) (earlier vocabulary:
  AgentRunner, runner-owned-by-flow) and folds in
  [`open-questions.md`](archived/open-questions.md). Start here.
- **Also supersedes** the two historical notes ([`loop-state-machine.md`](archived/loop-state-machine.md),
  [`multi-agent-orchestration.md`](archived/multi-agent-orchestration.md)). All four now live in `archived/`.
- **Captured:** through a long design conversation, 2026-05/06. Forward-looking;
  verify file/line references against current code before acting.
- **Not** the implementation plan — that's a separate follow-up.
- **Reading level — important:** the *conceptual model* (the four nouns, the
  splits, the layering) is settled. The concrete **interfaces** (Engine / Step /
  Agent / Backend / Compute / Flow signatures + the contracts + loop-state shape),
  the **concurrency execution model** (how one imperative flow body yields N
  concurrent trials), and the **failure / recovery model** (timeouts, retries,
  idempotency, crash recovery, cost caps) are **deliberately not specified here** —
  they are the first work of the implementation plan. **See §14 "must be designed
  before the build." Do not read this as build-ready.**

---

## 1. Goal — what this delivers

Turn the agent loop from prose-driven into a clean, extensible system whose
parts are obvious to a newcomer. Concretely it delivers:

- **Enforced operational guarantees** — `--max-iter`, stop conditions, budgets,
  in code (not prose the LLM is trusted to honor).
- **Cheap extensibility** — adding an agent or a flow is a small, local change.
- **Visibility** — orchestration is readable, testable, traceable.
- **Bounded AI** — each agent's **tools are its capability boundary** (§9).
- **A pluggable compute layer** — heavy training runs **local or in the cloud**
  (e.g. Saturn) behind one interface (§10).
- **Concurrency** — many trials in parallel to speed discovery, statelessly (§8).
- **Backend portability** — Claude Code / Codex / API are swappable backends.
- **Dual-mode** — autonomous (headless) *and* interactive.

---

## 2. The problem

Two complaints, one root cause:
- **`--max-iter` isn't enforced** — it's prose the LLM is trusted to honor.
- **Adding a role is a scavenger hunt** — the chain lives in `SKILL.md` prose and
  the role set is smeared across the library.

**Root cause:** orchestration (turn order, handoffs, stops, context routing)
lives *inside the LLM* (a long-lived "manager" session running prose) instead of
in deterministic code.

---

## 3. Core principle: judgment vs. control flow

Two separable things get delegated to "the agent":

| | What it is | Home |
|---|---|---|
| **Judgment** | *what* to do — the next hypothesis, the `model.py`, whether to stop | the **LLM** (irreducible) |
| **Control flow** | *what to do next* — sequencing, stops, retry, ordering | deterministic **code** |

The old system put both in the LLM, so control flow inherited no-guarantees /
no-tests / no-visibility. The fix: **pull control flow into code; leave judgment
in the agent.** Boundaries *enable* autonomy — code-owned rails let you safely
hand the agent a larger judgment space.

**The recurring move — split any unit that bundles judgment + a deterministic
effect:**
- the *manager* bundled orchestrate + judge → orchestration becomes code (the
  Engine), judgment becomes discrete agent calls;
- the *coder* bundled write + execute → the coder only writes `model.py`; code
  runs it;
- the *runner* (train+eval) is **pure deterministic execution** → it belongs in
  the ML library, not the agent layer (§5).

---

## 4. The mental model: building blocks vs. composition (sklearn / pytorch)

The clearest way to explain the structure — and the analogy to keep in the doc:

| | building blocks | the **composition** (thin) | the runtime runs it |
|---|---|---|---|
| **sklearn** | `StandardScaler`, `SVC`, `cross_validate` | `Pipeline([scaler, svc])` | `cross_validate(pipe, X, y)` |
| **pytorch (Lightning)** | `nn.Linear`, `optim.Adam`, `DataLoader`, `Trainer` | `class LitModel(LightningModule)` | `Trainer().fit(LitModel())` |
| **ours** | `agents`, `backends`, `engine`, `runner` | `class Improve(Flow)` | `engine.run(Improve())` |

So: **`Engine : Flow` :: `Trainer : LightningModule` :: `cross_validate : Pipeline`.**
The building blocks are reusable, single-purpose units; the **Flow is a thin
composition** that wires them, one level up. A flow doesn't reimplement blocks —
it references them. The **coupling lives in the composition** (a Pipeline couples
*this* scaler with *this* classifier; a flow couples *these* agents with *this*
runner), so the blocks stay independently reusable.

---

## 5. The architecture: two halves, four agentic nouns

The package is **two layers** (its own stated principle):

```
 DETERMINISTIC ML LIBRARY  (no AI — usable by a human; sklearn-like)
   data/  model/  eval/  runner/                 building blocks + trial execution
   mlflow/  trial/  experiment/  project/         state + lifecycle
   validate/  utils/  cli/

 AGENTIC SYSTEM  (the AI loop — exactly four nouns)
   engine/     the runtime: runs a flow, one run(step), budget, concurrency, state
   agents/     proposer · coder · investigator   (prompt + contract + tools)
   backends/   where any step runs:  agent/ (claude·codex)  ·  compute/ (local·saturn)
   flows/      improve.py · investigate.py        (thin compositions)
```

The **runner stays in the library** (it's deterministic ML execution, peer to
`data/model/eval`); the agentic layer only *calls* it. The coupling between the
coder and the runner is the **`BaseModel` contract**, not co-location.

**Dependency rule (the target, not today's state):** the agentic system *should*
depend only *down* on the library + substrates; only adapters (CLI / plugin)
*should* depend up. This is **not yet true** — `automl/__init__.py` re-exports
`Proposal` from `agent/`, and `validate/targets.py` imports
`agent.checks.proposal_schema` while `agent/checks.py` imports back from
`validate/` (a real cycle). Cutting those up-edges is part of the refactor (§12).

---

## 6. The entities (settled vocabulary)

- **Engine** — the runtime. Owns the **guarantees**: budget *enforcement*,
  checkpoints, telemetry, state I/O, and concurrency (via a configured
  `concurrent.futures` executor, §8). One generic operator: `engine.run(step)`.
  (The *flow* owns the loop's *sequence*; the engine owns the guarantees around
  it — §7.)
- **Agent** — a reasoning unit: **prompt + contract + tools**, plus a *shape* —
  a one-shot **workflow** agent (proposer, coder) or an open-ended **ReAct** agent
  (an investigator, which runs its *own* bounded, budgeted effect-loop). **Today
  only proposer and coder exist; investigator and others are planned.** Its
  *tools are its capability boundary* (§9).
- **Step** — the unit the engine runs. Two kinds, *one operator*: an **agent
  step** (run an agent) or a **run step** (execute a trial). The step
  encapsulates its kind; the flow author doesn't branch on it.
- **Backend** — *where a step runs.* Two families: **agent backends** (`claude`,
  `codex` — they run **on your machine**, since they need your login) and
  **compute backends** (`local`, `saturn` — a job, run locally or in the cloud).
  (`local` is the *name* of one compute backend; agent backends always run
  on-machine but aren't named "local.")
- **Runner** — deterministic execution of one trial (`fit(model.py)+data → eval
  → record`). Lives in the library. The "work" a run-step performs.
- **Flow** — a thin **composition** (a recipe), named by intent (`improve`,
  `investigate`). It *is* its registry entry. Imperative body (so the loop is
  dynamic). Like a `LightningModule`.
- **State** — `mlflow` (durable source of truth) + `gcs` (heavy bytes).
- **Contract** — a typed handoff owned by *its producer*: an **agent** (`Proposal
  | Stop`, from the proposer) or the **runner** (`TrialResult`). Consumers depend
  on the contract, not the producer.
- **Checkpoint** — *optional* pause point in a flow, with a pluggable handler
  (none / human / agent); which handler applies is chosen by **risk tier** —
  auto-pass low-risk, escalate high-risk to a human/agent (§7).

---

## 7. Execution model

### Generic `run(step)`, compute chosen once at launch

```
automl run improve --project X --max-iter 10 --max-concurrency 2 [--compute saturn]
```

`engine.run(step)` is generic and routes by step kind:
- **agent step** → always the **local** agent backend (auth lives here; never
  remote);
- **run step** → the **compute configured for this run** (`local` default, or
  `saturn`).

So the user picks compute *once*; "agent local, training remote" falls out
automatically. (A step may override the run-level default if ever needed.)

### Who calls whom (one iteration)

1. You run `automl run improve`. The **CLI adapter** starts the **Engine** with the
   `improve` **Flow**.
2. The Engine enters the flow's loop. Each iteration:
   1. **read** mlflow for context;
   2. **agent step** → the **proposer** (via the agent backend) → a `Proposal` (or `Stop`);
   3. optional **checkpoint**;
   4. **agent step** → the **coder** (via the agent backend) → it **writes `model.py`** into the trial dir;
   5. optional **checkpoint**;
   6. **run step** → the **compute backend** executes the **runner** (train → eval → record) → a `TrialResult` lands in **mlflow**.
3. Repeat until the engine's budget stops it or the proposer returns `Stop`.

(Both agent calls are **agent steps**; only step 6 is a **run step**.)

### The flow body (imperative — dynamic loop; Prefect-style, see §11)

```python
# flows/improve.py
class Improve(Flow):
    agents = [proposer, coder]
    runner = trial_runner
    def body(self, engine, session):
        while engine.budget_ok():                       # dynamic loop, NOT a static DAG
            ctx      = engine.context(session)          # reads mlflow
            proposal = engine.run(agent_step(proposer, ctx))
            if proposal.stop: break
            proposal = engine.checkpoint("proposal", proposal)   # optional handler
            trial    = engine.run(agent_step(coder, proposal))   # writes model.py
            engine.checkpoint("code", trial)                     # optional handler
            engine.run(run_step(self.runner, trial))             # → compute backend
```

**Who owns the loop:** the *flow body* owns the **sequence** (the `while`, the
order of steps); the *engine* owns the **guarantees** — `engine.budget_ok()` is
the flow *asking* the engine, which authoritatively enforces `--max-iter`, step
budgets, and consecutive-failure stops. Both are deterministic code, so
orchestration is out of the LLM regardless; the split is *sequence* (flow) vs.
*enforcement* (engine).

**Checkpoints, risk tiers & pause/resume:** a checkpoint's handler is chosen by
**risk tier** — low-risk auto-passes; higher-risk escalates to a human or a
reviewer agent. An interactive, human-handled checkpoint **pauses by persisting
to mlflow and exiting**; the next invocation **resumes** from that phase. The
resumable **loop-state** encodes `{iteration, phase, trial_dir}` (derived from
mlflow) — so there is no long-lived blocking process.

### Dual-mode (one logic path, thin doors)

| Door | Mechanism | Drives the loop |
|---|---|---|
| `automl run improve` (CLI) | calls the Engine | code — autonomous, guaranteed |
| `/automl run` (skill) | shells to the same CLI, **streams** output | code; human watches / reviews at checkpoints |
| `/automl propose` (skill) | runs a single step | the human sequences manually |

Skills are thin (~CLI calls); prompts live in the library; **the CLI is the
seam** between the Claude-Code-plugin world and the Python library. Distribution
is two artifacts: `uv tool install` (lib+CLI) and a Claude Code plugin install.

---

## 8. Concurrency

The Engine runs steps through a configured (constructor-injected)
**`concurrent.futures` executor** — the stdlib wheel, *not* a hand-rolled scheduler.
The *aim* is that one flow expresses both serial and parallel intent; the concrete
mechanism that makes one **imperative** flow body actually yield N concurrent
trials (the engine drives the pool vs. the body submits-and-awaits) is **not yet
specified — see §14.** Heavy work (claude/codex, training) is all subprocess/job
I/O, so a thread/async pool fits *for the waiting* — **but a hung trial cannot be
cancelled in-thread**; enforcing a per-trial timeout needs a **process/kill
boundary** (§14), which `concurrent.futures` alone does not give you.

- **`max_concurrency`** = throughput (how many at once); **`max_iter`** = budget
  (how many total). Orthogonal.
- **Continuous pool:** keep `max_concurrency` trials in flight; as each finishes,
  launch the next — informed by all *completed* trials (read from mlflow).
- **State, stated honestly:** each trial's *execution* is stateless (reads mlflow
  + writes its own run). The *orchestration* is **not** fully stateless — pool
  admission, the total-`max_iter` counter, and the resume cursor are coordinated
  state. We keep them minimal and mlflow-derived, but the earlier "fully stateless"
  framing was too strong, and resuming a *partial pool* (some done, some in-flight,
  killed mid-run) is a real reconciliation problem (§14).
- **Duplicate risk only on *simultaneous* launches** (cold start, or two
  finishing together). Handle it by **diversifying at generation**: one quick
  "direction" call produces *K orthogonal directions*, and a **Jinja proposer
  template** injects a different direction per parallel proposal (single launches
  inject none). A cheap distinctness check guards the batch.
- **Parallel also needs:** trial isolation (already per-trial dir + per-trial
  mlflow run ✅) and a **concurrency-safe session lock** (today one-per-route →
  must allow N) — bounded work, scheduled when parallel lands.

Don't adopt LangGraph/Temporal/Ray/Airflow for this — stdlib executor + a thin
engine. Revisit only at real graph/distributed scale.

---

## 9. Agent tools / capability layer

Governance is modeled the standard way: **an agent's tool-set *is* its capability
boundary** (as in Claude Code / Codex / MCP / Goose). Read-only by default;
narrow explicit grants; the absence of a tool is the boundary. Enforced by the
agent backend.

- *proposer:* `read_mlflow` — **no** write tool.
- *coder:* `read_project` + `write_model_py` (its trial dir only) — no mlflow.
- *investigator:* read a run's artifacts + data — no writes to production.

Symmetrically, a read-mostly agent's *output* goes to a **non-production sink**
(scratch / a findings artifact / just-returned), never production mlflow — the
sink is the other half of its boundary. And every agent loop — including a ReAct
investigator — runs under a **step budget**, the same way the modeling loop runs
under `--max-iter`.

This layer is reused across agents (the investigator/incident agents lean on it
hardest). It lives *with the agent definition*, so an agent's powers are legible
where the agent is defined.

---

## 10. Compute layer

`backends/compute/` — one uniform interface, two implementations:

```
backends/compute/
├── base.py     Compute.run(trial) -> Result        # uniform
├── local.py    run the runner in a local subprocess (today)
└── saturn.py   package → submit job → await → result   ← all Saturn packaging lives here
```

`engine.run(run_step)` just calls `compute.run(trial)`; the *flow* never names a
backend. But the **engine is not fully Saturn-agnostic**: `local.py` *returns* a
`TrialResult`, while `saturn.py` submits a job and the result comes back **through
mlflow** (poll until it appears). The two backends differ in liveness and failure
— a cloud job can die without ever writing mlflow → infinite poll — so the remote
path needs a **job handle + liveness/timeout** the local path doesn't (§14).

`saturn.py` ships the **trial** (`model.py` + shim) + **credentials** (`.env`:
GCS read, mlflow write) + an **entrypoint** that invokes the **runner**, on top
of a **pre-built pinned image** that already has the deterministic library at
locked versions. The remote worker runs the runner and logs to the *same*
mlflow; the result returns *through mlflow*.

> **Load-bearing constraint:** the cloud image must be the **same pinned
> environment** as local — the shared deployment Docker image — never a fresh
> dependency resolve. Otherwise you reintroduce the cloudpickle/version-drift bug
> (`CLAUDE.md`, commit `d4a9598`). Identical env is what makes a remote trial
> bit-for-bit equivalent to a local one — the whole point of moving the heavy
> step off the box.

Two more things the cloud path must commit, or it breaks package invariants:
(1) the Saturn image is the **same** shared, pinned/serving image — not a second
image kept byte-identical by hand (and mind credential/entrypoint pollution of the
serving image); (2) the remote worker loads training data through the **same
`DataPipeline` override / SQL-canonical seam**, not a new GCS-parquet path (a
third, unsanctioned data entry point). Serving-parity validation
(`runner/serving_validation.py`) also needs a home in this picture (§14).

---

## 11. The flow as a composition (Prefect, not Airflow)

A flow's body is **imperative Python** (so the loop is dynamic — it runs until
the proposer stops, and each step depends on the last result). That rules out a
static **Airflow** DAG. The right teacher is **Prefect/Dagster**: a flow is normal
code, but each *step* is a first-class, observable unit the runtime wraps (retry,
telemetry, budget, concurrency).

| Pipeline-framework concept | Ours |
|---|---|
| a task/operator | `engine.run(agent_step(...))` / `engine.run(run_step(...))` |
| the task's definition | the **agent** (prompt+contract+tools) / the **runner** |
| connections | the imperative wiring in the flow body |
| definition vs. execution | the **flow** defines; the **engine** runs |
| scheduler/executor | the **engine** (concurrency, budget, checkpoints) |

Take the *patterns* (steps-as-units, definition-vs-execution, observability); do
**not** adopt the heavyweight framework. Built-in flows live in `flows/`; a
project may author its own (like writing your own Pipeline).

---

## 12. Repo layout

```
automl/
│  ─ deterministic ML library (exists today; no AI) ─
├── data/ · model/ · eval/ · runner/             building blocks + trial execution
├── mlflow/ · trial/ · experiment/ · project/     state + lifecycle
├── validate/ · utils/ · cli/
│
│  ─ agentic system (the only new part — 4 nouns) ─
├── engine/        the runtime (run(step), budget, concurrency, checkpoints, telemetry)
├── agents/        proposer · coder · investigator   (prompt + contract + tools)
├── backends/      agent/ (claude·codex) · compute/ (local·saturn)
└── flows/         improve.py · investigate.py        (thin compositions)
```

The plugin layer (`skills/`, `agents/*.md`, `hooks/` at repo root) stays thin and
shells to the CLI. (A "noun" can be a file or a folder by size — `engine.py` may
be a file until it grows.)

**Migration note (verified against current code — harder than a rename):**
- Today's `agent/` (singular) is a *mix*: `launch.py` (the `claude` subprocess
  builder → `backends/agent/`), `proposal.py` (the contract → `flows/improve` or
  `agents/`), `checks.py` (proposal validation — see the cycle below), and
  `timeline/` + `proposer_context.py` which are **deterministic library** code
  (they read mlflow/data) and redistribute to the library / engine telemetry,
  *not* to `agents/`. So `agent/` doesn't cleanly "become" the agentic layer.
- `runner/` (~2,200 lines) does **not** cleanly bisect. `trial.py` is roughly the
  chain (→ library `runner/`), but `session_lock.py` is *coordination* (CLI-facing
  → an engine/coordination home) and `serving_validation.py` (~590 lines, spawns a
  subprocess for serving-parity) is *trial-time validation* — neither is "the
  chain" nor "compute substrate." Each needs a deliberate home.
- **Proposal-validation is a *cycle*, not a one-way seam:** `validate/targets.py`
  imports `agent.checks.proposal_schema` **and** `agent/checks.py` imports back
  from `validate/base.py`. Untangling it (plus the `automl/__init__.py`
  `Proposal` re-export) is what restores the §5 dependency rule.
- The `trial` run.py-template embeds `from automl import runner`
  (`trial/template.py`) — a real coupling to decouple.
- `investigator` is **not built yet** (only `proposer`/`coder` exist); it's
  forward-looking throughout this doc.

- **This redraws the contributor guide's two-layer line.** `CLAUDE.md` today puts
  the judgment parts (prompts, agent defs) in the *plugin* layer (`skills/ agents/
  hooks/`) and keeps `automl/` deterministic. Moving **prompts into the library**
  (`automl/agents/`) and thinning skills to CLI shells deliberately **supersedes
  the guide's tier table and its "judgment → plugin" split** — and clashes by name
  with the plugin-root `agents/*.md` (Claude Code subagent files, which become
  thin/optional). `CLAUDE.md` must be updated when this lands.

This absorbs the `multi-runner-architecture.md` runner split. The detailed
sequencing belongs in the implementation plan.

---

## 13. Decision log

| # | Decision | Why |
|---|---|---|
| D1 | Orchestration in **code**, not prose | guarantees + visibility + tests; resolves both root problems |
| D2 | The **runner is deterministic library**, not agentic; the flow *calls* it; coupling = `BaseModel` contract | de-bloats the agent layer, un-tangles the flow, restores the two-layer principle |
| D3 | **One `run(step)` operator**; agent-vs-compute is a step kind, hidden from the flow | Prefect-style uniform task; smaller engine API. *(One-shot agents: the engine runs the effect step. **ReAct** agents run effects themselves, inside their own bounded/budgeted loop — their shape.)* **Caveat:** uniform for *dispatch*; the two kinds *fail* differently (retry/timeout/idempotency), so the engine's failure model is step-kind-aware (§14). |
| D4 | **Compute chosen once at launch**, routed by step kind (agent→local, run→configured) | "agent local / training remote" for free; user picks once |
| D5 | Backends behind ports: **agent backends** (claude/codex) + **compute backends** (local/saturn) | Codex-swap + cloud execution; both pluggable |
| D6 | Handoffs are **typed contracts owned by their producer** (an agent, or the runner for `TrialResult`); depend on contract not producer | safe insert/swap; non-modeling shapes possible |
| D7 | **Tools = the capability boundary** (per agent, read-only default) | standard (MCP/CC/Goose), intuitive, enforces scope |
| D8 | Concurrency via **stdlib `concurrent.futures`** (not a framework) | reuse the wheel; *but the execution model that makes one imperative flow body parallel is a must-design item (§14-C), and a hung trial needs a process/kill boundary (§14-D)* |
| D9 | Diversify parallel proposals **at generation** (direction-gen + Jinja), only on simultaneous launches | kills duplicates; keeps *trial execution* stateless (orchestration still has a small coordinated state — §8) |
| D10 | **Flow = thin composition** (`Pipeline`/`LightningModule`), imperative body | dynamic loop; coupling lives in the recipe; extensible |
| D11 | **mlflow stays the durable state**; loop-state encodes phase, derived from mlflow | keep the one thing already right |
| D12 | Skills thin, prompts in library, **CLI is the seam**; two distributable artifacts | enforces "skill = glue" |

---

## 14. Open questions (updated)

Resolved since the first pass: search-strategy-as-a-layer (**no** — the agentic
proposer + narrow model space make it moot); capability layer (**tools**, D7);
compute (**a layer**, §10). *(Concurrency's **concept** is settled but its
execution model is **not** — see C below; "executor swap resolves it" was too
strong.)*

### Must be designed *before / early in* the build (NOT deferrable)

The reviews surfaced that this doc settles *concepts*, not the *operational layer*.
These are the specs the first commits need — the top of the implementation plan:

- **A. Interface stubs.** Write the `Engine` (`run`/`budget_ok`/`context`/
  `checkpoint` + constructor), `Step` (+ `agent_step`/`run_step`), `Agent`,
  agent-backend `run`, `Compute.run`, `Flow`, the contracts, and the loop-state
  shape — porting `design.md`'s surviving shapes into the new vocabulary.
- **B. `run(step)` vs `run(flow)` + sync-vs-Future.** Disambiguate the two `run`s;
  decide value-vs-`Future` (§7's body reads results synchronously, §8 needs N in
  flight). *The* load-bearing unresolved decision.
- **C. Concurrency execution model.** How the imperative body yields a pool; an
  atomic total-`max_iter` counter; partial-batch-failure semantics; and a real
  spec for "diversify on simultaneous launch" (direction step + Jinja + distinctness).
- **D. Trial timeout / kill.** A **process boundary** (subprocess + SIGKILL, like
  today's `run.py`) so a hung `fit()` is reclaimable — `concurrent.futures` can't.
- **E. Failure & recovery model.** A failure *taxonomy* (agent / trial / transient-
  cloud) with per-class retry/backoff; the **failed-trial → proposer feedback
  loop**; crash recovery (reconcile stale `RUNNING` runs, clean orphaned jobs).
  Build on `runner/failures.py` (`RunnerFailureReport`).
- **F. Idempotency.** Keys for run-steps so retry/resume don't duplicate runs
  (Saturn "submit→await→result-via-mlflow" is at-least-once).
- **G. Cost controls.** Token/$ + wall-clock caps beside `--max-iter`; honor the
  existing `--time-budget` flag.
- **H. Testing seam.** A **fake agent backend** + recorded-proposal fixtures — the
  whole premise is testability.
- **I. N-safe session lock + its home.** Counting semaphore + per-holder liveness
  (`runner/session_lock.py` is one-per-route today).
- **J. `validate ↔ agent` cycle** + the `automl/__init__.py` `Proposal` re-export —
  prerequisite for the §5 dependency rule.
- **K. Contract versioning.** Reading historical mlflow runs across a contract
  change (no-backcompat covers config, not durable-state reads).
- **L. Saturn image-equivalence + data seam** (§10), `serving_validation` home, and
  the **three-tier / two-layer reconciliation** + `agents/` name clash (§12).

### Deferrable (build when demanded)

1. **Graph topologies** — only if branching/parallel beyond the loop demands it.
2. **Observability** — a system-level view beyond per-trial mlflow; the engine is
   the natural telemetry collection point.
3. **Always-on / event-driven** — schedule/drift triggers as another inbound adapter.
4. **Sandbox for compute** — likely overkill now.
5. **Governed learning store** — REA shipped one; reconsider priority.
6. **Security / trust layer** — prompt-injection / PII / agent identity; real once
   the agent reads production data or goes conversational.

---

## 15. Industry validation

This design is **convergent**, and "execution boundary" has an industry name —
the **harness** ("Agent = Model + Harness"; ~70% of agent performance is *outside*
the model, so harness engineering beats model-tier upgrades — the case for this
whole refactor).

- **Meta KernelEvolve** literally uses a **"job-harness"** that evaluates each
  candidate and feeds rich diagnostics back to the LLM — our judgment/execution
  split, named. (Take: feed *rich diagnostics* in `TrialResult`, not just a metric.)
- **Meta REA** runs the ML-experimentation lifecycle (hypothesis→train→debug→
  iterate), **scoped to one codebase**, **compute budgets confirmed up front and
  halted at threshold** (= code-enforced `--max-iter`), with **plan-approval up
  front** and **risk-tiered auto-handling**. (Takes: a plan checkpoint; risk-tiered
  handlers.)
- **Uber** — LangGraph orchestration, **grounds the LLM in deterministic tools**,
  and an **encapsulated interface** domain experts extend without AI internals.
- **Block Goose** — any-LLM **provider abstraction** + desktop/CLI/API surfaces
  (our backends + three doors).
- **Google self-evolving recsys** — offline (proxy metrics) / online (business
  metrics) split — our dry-run/production split.

---

## 16. Risks & non-goals

**Risks:** schema break (project config grows a flow/agent registry — fine under
no-backcompat); cold-start cost (fresh agent sessions re-ingest context — small
vs. minutes of training); **harness co-evolution** (prompts tuned to one backend
may degrade on another — the swap abstracts invocation, not prompt-fit); the
**pinned-env constraint** for cloud compute (§10).

**Non-goals:** the implementation plan (separate doc); the governed learning store
(deferred seam); a heavyweight orchestration framework; real-time/online ML,
per-model serving envs, cross-project learning, emergent multi-agent negotiation
(deployment-policy / determinism boundaries). **The agent stops before
deployment** — even Meta's REA is scoped to experimentation.

---

## 17. Glossary

- **Engine** — internal runtime; runs a flow; one `run(step)` operator.
- **Step** — the unit the engine runs: an *agent step* or a *run step*.
- **Agent** — a reasoning unit (prompt + contract + tools); tools = its boundary.
- **Backend** — where a step runs: *agent* (claude/codex, local) or *compute*
  (local/saturn).
- **Runner** — deterministic execution of one trial (library, not agentic).
- **Flow** — a thin composition (recipe) wiring agents + a runner into a
  sequence; named by intent; ≡ its registry entry; like a `Pipeline` /
  `LightningModule`. (Stopping comes from the engine's budget + the proposer's
  `Stop` contract, not a flow-owned field.)
- **Shape** — whether an agent's path is known (*workflow*: proposer→coder) or
  open-ended (*ReAct*: investigator, which runs its own bounded effect-loop).
- **Compute** — the substrate for run-steps (local/cloud); chosen once per run.
- **Contract** — a typed handoff owned by its producer (an agent, or the runner
  for `TrialResult`).
- **Checkpoint** — optional pause point with a pluggable handler (none/human/agent).
- **State** — mlflow (durable) + gcs (heavy bytes).
- **Harness** — industry term for the execution boundary around an LLM; our
  Engine + backends + runner are it.
