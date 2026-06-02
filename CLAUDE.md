# brigit-automl — contributor guide

> **This is the current shape of the system — high-level, and expected to
> evolve.** Treat it as the map a developer needs to start working, plus the
> few decisions we don't want re-litigated. For exact behavior the code is the
> source of truth. Users want [`README.md`](README.md).

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

## The design

The library is layered by **dependency direction** — what imports what. From
the bottom up:

- **Leaves** — `utils` and `errors` depend on nothing internal; anything may use
  them.
- **Persistence seam + core domains** — `mlflow` (the single place that talks to
  MLflow) together with `data`, `eval`, `project`, and `trial`. The domains ride
  on the seam *and* the seam reaches back into them, so they form a co-dependent
  core rather than a clean stack (a few imports are deferred to break load-time
  cycles). `model` sits just above, on `project` + `validate`.
- **Orchestration** — `runner` (executes one trial) and `agent` (drives the
  loop) sit high: each pulls in most of the domains to do its job.
- **Surface** — the `automl` CLI and the agent skills sit on top; a skill
  reaches the library only through the CLI, never directly.

**The nouns** are the shared vocabulary — one home per concept: **Project,
Dataset, Experiment, Trial, Proposal, Model**. (The current set; it may grow.)

**Three places code can live**, decided mechanically:

- **The library** (`automl/`) — all reusable logic and the contracts; the source
  of truth. Anything reused across skills, notebooks, or tests goes here.
- **The CLI** (`automl/cli/`) — a thin layer that parses arguments and calls the
  library. No logic lives here.
- **Skill glue** (`agent-skills/skills/<name>/scripts/`) — a small
  adapter that assembles one skill's context from the library.

**Validation draws boundaries, kept light.** Each domain defines its own
contract (what a valid project, model, or proposal is); validation checks
against those contracts at the edges rather than scattering defensive checks
throughout.

## Folder structure

```
brigit-automl/
├── automl/         # the library: cli/ + the domain packages (data, model, eval,
│                   #   experiment, trial, agent, project, runner, validate,
│                   #   mlflow seam, utils)
├── agent-skills/   # the Claude Code plugin (its own root): skills/ · agents/ · hooks/ ·
│                   #   references/ + .claude-plugin/plugin.json. Loaded via --plugin-dir.
├── projects/       # one folder per problem: the recipe (config.py) and its assets
├── docs/           # design & planning notes — see docs/README.md
└── tests/          # unit / integration / contracts / e2e
```

The two halves are the **library** (`automl/`) and the **agent layer** that
drives it — a self-contained Claude Code plugin under `agent-skills/` (skills,
subagents, hooks, references, plus its own `.claude-plugin/plugin.json`). The
AutoML loop loads it automatically via `--plugin-dir`; for interactive skill use,
run `claude --plugin-dir agent-skills` (or symlink it into `~/.claude/skills/`
for flag-free auto-load). Each problem you work on is a self-contained recipe
under `projects/<name>/`.

## What we don't want to break

These are settled. Flag them with the user before working around one — don't
relitigate them in passing.

- **`uv` for everything.** Never `pip`, never the system Python.
- **MLflow and GCS are the only sources of truth** — MLflow for durable state,
  GCS for the heavy bytes (datasets, artifacts). Don't invent a local file as
  state.
- **One model packaging contract (`cloudpickle`) on one shared serving image** —
  safe only because training and serving environments are identical. No other
  formats, no per-model environments.
- **SQL / warehouse data is the canonical entry point.** Project recipes may use
  `DATA.source` or a `DataSpec.pipeline_cls` override for examples and harnesses,
  but those are explicit project-owned escape hatches.
- **Do not treat example harnesses as the product.** `projects/example_homecredit/`
  is an end-to-end example and regression harness; AutoML is the artifact.
- **The library is the source of truth; the CLI stays thin.** Reusable logic
  lives in `automl/`, never in a skill or a CLI wrapper.

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
