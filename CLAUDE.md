# brigit-automl — contributor guide & design

> **One document, two jobs:** the durable design — the decisions that shape
> the system and the reasons behind them — plus the map a developer needs to
> start working. The principles change rarely; the specifics they govern
> (today's module list, today's noun set) evolve with the code. For exact
> behavior the code is the source of truth.

## Scope

This is the package-level guide. Paths are relative to this repo root. A
parent-directory `CLAUDE.md` may exist on some local development machines with
machine-specific setup such as MLflow, credentials, or test-harness folders; it
is optional local context, not a source of package conventions.

## What we're building

brigit-automl is an **AutoML harness**: a standardized workflow for building
tabular ML models faster. A prediction problem is described once as a project
recipe, and the harness runs a repeatable loop that proposes, implements, and
evaluates model trials — recording everything so results are reproducible and
comparable instead of living in throwaway scripts.

## The shape of the repo

```
brigit-automl/
├── automl/         # the core library: all reusable logic and the contracts;
│                   #   cli/ + the domain packages (data, model, eval, experiment,
│                   #   trial, agent, project, runner, validate, mlflow seam, utils)
├── agent-skills/   # the agent layer: a self-contained Claude Code plugin
│                   #   (skills/ · agents/ · hooks/ · references/ +
│                   #   .claude-plugin/plugin.json). Loaded via --plugin-dir.
├── projects/       # the working directory: one folder per prediction problem —
│                   #   the recipe (config.py), overrides, notebooks, assets
├── tests/          # unit / integration / contracts / e2e
├── docs/           # optional: lifecycle tracking of efforts (to-do / execution /
│                   #   archive) — see docs/README.md. Doesn't affect the system.
└── CLAUDE.md       # this guide — design principles + developer notes
```

The first three are the system: the **core library**, the **agent layer**
that drives it, and the **projects working directory** where users create a
project and work from. The rest is scaffolding around them — `tests/` keeps
the system honest, `docs/` is an optional structured way to track what's
next, in flight, and done.

The AutoML loop loads the plugin automatically via `--plugin-dir`; for
interactive skill use, run `claude --plugin-dir agent-skills` (or symlink it
into `~/.claude/skills/` for flag-free auto-load).

## Design principles

These are the settled decisions and why we made them. Flag it with the user
before working around one — don't relitigate them in passing.

### Surfaces are thin

The CLI and the agent skills are consumers of the library — they contain no
business logic of their own. There are three places code can live, decided
mechanically:

- **The library** (`automl/`) — all reusable logic and the contracts; the
  source of truth. Anything reused across skills, notebooks, or tests goes
  here.
- **The CLI** (`automl/cli/`) — parses arguments and calls the library,
  nothing else. If a CLI verb needs logic, the logic moves into the library
  and the verb calls it.
- **Skill glue** (`agent-skills/skills/<name>/scripts/`) — a small adapter
  that assembles one skill's context. A skill reaches the system only
  through the CLI, never by importing the library directly.

**Why:** one home for behavior means one place to test it and one place to
fix it, and it keeps the surfaces replaceable — a new CLI or a different
agent harness can be swapped in without changing what the system does. The
moment logic leaks into a wrapper, the wrapper becomes load-bearing and
stops being replaceable.

### Generic core, project-owned specifics

The library stays **generic**: it knows what a project, dataset, or model
*is*, never what any particular one looks like. Everything specific to one
prediction problem lives in that problem's folder under `projects/`:

- **The recipe** (`config.py`) describes the problem once — the dataset, the
  task, the evaluation — in the contracts the library defines.
- **Overrides** are project-owned code: a custom pipeline, a project model,
  preprocessing, a custom evaluation. The library defines the contract and
  the default; the project may supply its own implementation.
- **Notebooks and assets** for working the problem live alongside.

Data follows the same rule. For internal projects, the **warehouse
(Snowflake) is the canonical entry point** — a dataset is a snapshot of
warehouse base tables, defined in the recipe. Other entry points
(`DATA.source`, a `DataSpec.pipeline_cls` override) exist as **explicit,
project-owned escape hatches** for examples and harnesses — not as parallel
paths into the system.

**Why:** the generic/specific split is what lets one library serve every
problem without forking. When a project needs something unusual, it gets an
explicit home inside that project instead of a special case inside the core
— so the core never accumulates per-project knowledge, and a project folder
tells you everything that makes its problem different.

### Layered by dependency direction

Like sklearn or PyTorch, the library is organized low → mid → high, and the
layering is defined mechanically by **what imports what** — imports point
downward, never up. Two rules keep it honest:

- **No cross-cutting concerns.** A concern lives at one altitude, in one
  place. If two domains need the same thing, it moves *down* into a shared
  lower layer — it doesn't get duplicated sideways.
- **Each module stays in its lane.** A package owns its responsibility and
  nothing else; reaching into a peer's internals is a smell that the
  boundary is drawn wrong.

**Why:** dependency direction is the one structural rule a codebase can
enforce mechanically. When imports only point down, any module can be
understood, tested, or replaced knowing only the layers beneath it — and the
import graph itself documents the architecture.

Today's layers, from the bottom up:

- **Leaves** — `utils`, `errors`, and `validate` (the shared check-result
  vocabulary) depend on nothing internal; anything may use them.
- **Persistence seam + core domains** — `mlflow` (the single place that talks to
  MLflow) together with `data`, `eval`, `project`, and `trial`. The domains ride
  on the seam *and* the seam reaches back into them, so they form a co-dependent
  core rather than a clean stack (a few imports are deferred to break load-time
  cycles). `model` sits just above, on `project` + `validate`.
- **Orchestration** — `runner` (executes one trial) and `agent` (drives the
  loop) sit high: each pulls in most of the domains to do its job.
- **Surface** — the `automl` CLI and the agent skills sit on top; a skill
  reaches the library only through the CLI, never directly.

The known exceptions to the clean stack are debts we pay down iteratively —
the principle is the direction of travel, not a claim that we've arrived.
A dated snapshot of the measured import graph lives in
[`automl/ARCHITECTURE.md`](automl/ARCHITECTURE.md); like any doc it can go
stale — the code is the source of truth.

### A shared vocabulary of nouns

The system is described by a small set of nouns, and **each noun has exactly
one home**: currently **Project, Dataset, Experiment, Trial, Proposal,
Model**. The set will grow as the system grows; the principle is that when a
concept earns a name, it gets one home package, and everything else refers
to it there rather than redefining it.

**Why:** shared nouns are what let the library, the CLI, the skills, and the
humans talk about the same thing. Two definitions of "trial" is how
contracts drift apart silently.

### Durable state: two stores, three levels

Everything durable lives in exactly two places: **MLflow is the durable
record** (runs, params, metrics, tags, learnings) and **GCS holds the heavy
bytes** (datasets, artifacts). Nothing else is a source of truth — no local
file, cache, or side database ever holds state that can't be reconstructed
from those two.

Within MLflow, the record is organized in a strict hierarchy — one project,
many experiments, many trials per experiment — and **each level records a
different kind of knowledge**:

- **Project** — consolidated learning. The slow-moving picture of what has
  worked and what hasn't across experiments; the place a finding lands once
  it has become a trend.
- **Experiment** — the data snapshot it ran against, the EDA on it, and what
  was learned within that experiment.
- **Trial** — one concrete run, mostly auto-logged: code, params, metrics,
  artifacts.

Knowledge propagates **upward**: a trial result that repeats becomes an
experiment-level learning; an experiment-level learning that holds across
experiments is promoted to the project.

MLflow natively gives only two levels (experiment → run), so the hierarchy
is built by convention: our experiments are MLflow experiments namespaced by
route (`<project>/<experiment>`), our trials are runs inside them, and the
project level is a dedicated `<project>/000_overview` experiment that acts
as the project-level entity.

Transient QA and development runs must use a namespace beginning with
`qa/`, for example `qa/notebook-e2e-20260605` or
`qa/agent-timeline-hooks-20260605`. That applies to automated e2e tests,
manual one-off verification, and agent/dev experiments. The
`project delete --scope qa` cleanup command archives active QA namespaces
wholesale; putting temporary runs outside `qa/` makes cleanup manual and easy
to miss. Use the default
project namespace only for state the user intentionally wants to keep. Do
not create new `qa-*` namespaces; that older prefix remains cleanup-compatible
only so legacy artifacts can still be swept. For notebook QA runs, set
`AUTOML_NOTEBOOK_NAMESPACE=qa/<purpose>-<stamp>` before executing the
notebooks; the e2e notebook harness defaults to that shape and rejects
non-`qa/` overrides.

Cleanup is production-safe by default. `project delete` and `experiment
delete` archive the route first, then soft-delete the renamed MLflow
experiment: MLflow moves to `deleted/<original-route>`, GCS bytes move to the
matching `deleted/...` prefix, and local experiment state moves under
`projects/<name>/experiments/deleted/...`.
That frees the original MLflow name without requiring database access. `trial
delete` uses the same archive prefix for run-scoped GCS/local artifacts, then
soft-deletes the MLflow run. Permanent cleanup is a second explicit step:
`mlflow purge <deleted-route>` purges one archived route, `mlflow purge --scope
qa` purges archived QA routes, and `mlflow purge --scope deleted` purges all
archived routes. Purge is for local/admin contexts where `mlflow gc` and the
backend/auth stores are reachable.

**Why:** two well-known stores mean any machine or session can reconstruct
the full picture by pointing at them — results survive the laptop they were
produced on, and there is never a "which copy is right?" question. And
within the record, every piece of knowledge has a right altitude: pin the
data snapshot at the experiment and all its trials are comparable; keep
consolidated learning at the project and the agent loop starts each
experiment already knowing what the project has learned, without re-reading
every trial that came before.

### Validation draws the boundaries, kept light

Each domain defines its own contract — what a valid project, proposal, or
model is — and validation checks against those contracts at the edges:
where input enters the system and where one layer hands off to another. We
don't scatter defensive checks through the middle of the code.

The split follows the layering: `automl/validate` is a leaf holding only the
shared vocabulary (`Issue`, `ValidationReport`, `run_check`); each domain owns
both its checks *and* its recipe — which checks make a full validation, in
what order — in its `checks.py` (`model.validate_model`,
`project.validate_project`, `agent.validate_proposal`). Surfaces that need
every target (the CLI's `validate` verbs, the runner) import the recipes from
the domains; nothing imports upward.

**Why:** contracts at the edges catch bad state where it's cheapest to
explain, and keep the interior code free to assume its inputs are valid —
which is what keeps the domains small.

### Also settled

Smaller decisions that don't need a section but shouldn't be relitigated:

- **One model packaging contract (`cloudpickle`) on one shared serving image** —
  safe only because training and serving environments are identical. No other
  formats, no per-model environments.
- **`projects/example_homecredit/` is the entry example** — a complete
  problem for learning the workflow end-to-end, doubling as the regression
  harness. The product is the harness itself; the example exists to exercise
  and demonstrate it.
- **Forward-only evolution (for now).** No back-compat shims, no state
  migrations: old logged state is disposable — re-run rather than migrate.
  This is the pre-rollout posture; once others depend on the system, expect
  to revisit it and make the change loud here.

## Technical preferences

How we prefer to work with the system — just as settled as the principles
above, but about ergonomics rather than the system's shape.

### The session container: auto-load, override explicitly

Work happens inside a project/session container that holds the shared
context — which project is active, its config, where its state lives. The
container is **auto-loading**: it discovers and reads the active project so
everything downstream shares the same context by default. And it is
**overridable**: any piece can be replaced explicitly when a caller needs
to.

**Why:** sharing by default eliminates the "which config was that run
using?" class of drift — every notebook, CLI call, and skill in a session
sees the same world. Making overrides explicit keeps the escape hatches
visible instead of ambient.

### One toolchain: uv

Everything runs through `uv` — never `pip`, never the system Python. One
lockfile, one venv, one way to invoke anything (`uv run ...`).

**Why:** a single resolver and lockfile means every contributor, notebook,
CI job, and agent session executes in the same environment. Environment
drift is the least interesting way for an ML result to stop reproducing, so
we remove it by construction.

## Tests

`uv run pytest tests/<tier>/`:

- **`unit/`** — fast, no live services. **`integration/`** — multiple modules,
  may use file-backed MLflow.
- **`contracts/`** — ratchet tests that pin the shape (skills, CLI verbs,
  retired paths, doc phrases). **Change the shape → update the matching contract
  in the same change.** That's the point of them.
- **`e2e/`** — live services; skips unless `AUTOML_E2E=1` (or
  `AUTOML_E2E_NOTEBOOKS=1` for notebooks) and credentials are set.

`projects/example_homecredit/` is the bundled example to run end-to-end against.

## When in doubt

Ask. The **code is the source of truth for current behavior; the user is the
source of truth for intent.** `docs/` captures intent and history (see
[`docs/README.md`](docs/README.md)), not how things work today.
`docs/HANDOFF.md` is written only when wrapping a session for handoff (or
when asked) — never updated mid-session as a status log.
