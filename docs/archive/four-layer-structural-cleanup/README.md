# 🚧 Active Execution — Four-Layer Structural Cleanup

**This folder is the single home for work we are *currently executing*.** It is
deliberately separate from the design/spec history under
`docs/superpowers/automl-refactor/` (which will be retired post-merge) and from
forward-looking notes under `docs/to-do/`. If you are an agent or a human
resuming this work, **start here.**

## Resume in a new session — paste this

> *"We're executing the four-layer structural cleanup. Read
> `docs/execution/README.md` (including the Self-Driving Execution Protocol
> below) and `docs/execution/STATUS.md`, then drive the waves forward per that
> protocol: execute the pre-approved Wave A autonomously, then for each later
> wave author its just-in-time plan and **pause for my approval before executing
> it** (rule 4a), running task-by-task within an approved wave with targeted
> commits. Update `STATUS.md` at wave gates, handoffs, and blockers only. Run the
> baseline check first and report before changing anything. Also pause to raise a
> concern or get a decision you can't make from code/tests (rule 5)."*

## Self-driving execution protocol

The executing session drives itself **one wave at a time, task-by-task**, using
the **`superpowers:subagent-driven-development`** skill (a fresh subagent per
task with a review checkpoint between tasks). Rules:

1. **Find your place & author missing plans.** Read `STATUS.md` for the current
   wave + next unchecked task. **Only Wave A is detailed *and* user-reviewed
   today;** Waves B–E are scope-only. Before executing a scope-only wave,
   **author its bite-sized detailed plan with the `superpowers:writing-plans`
   skill** (append it under `cleanup-plan.md`) — then get it approved per rule 4a
   before executing. Never execute a wave from scope alone.
2. **Baseline first.** Run `uv run pytest tests/unit tests/contracts -q` and
   report the result before editing anything. (Wave A needs no MLflow/GCS/`.env`;
   later waves do — set them up when their plan calls for it.)
3. **Per task, in plan order (no skipping/reordering):** dispatch a subagent to
   do exactly that task's steps → run the task's tests → review the diff →
   commit with a **targeted `git add <specific files>`** (never `git add -A`).
   Do not update `STATUS.md` per task unless stopping mid-wave; summarize task
   commits and evidence at the wave gate.
4. **Move forward, with a plan-review checkpoint per wave.**
   - **4a — Plan gate:** before executing any wave whose detailed plan the user
     has **not** approved (every wave except A), STOP, post the authored plan
     (its path + a short summary of tasks and the wave's key risks), and wait for
     the user's explicit "go." **Wave A is pre-approved — execute it without this
     pause.**
   - **4b — Within an approved wave:** execute task-by-task autonomously
     (rule 3); do not pause between tasks except per rule 5.
   - **4c — At the wave's acceptance gate:** commit, mark the wave complete in
     `STATUS.md`, then author the next wave's plan and return to 4a. Continue
     through Wave E. Each wave boundary is a durable committed checkpoint.
5. **Pause only to raise a concern or get an intent decision — never guess.**
   Keep moving on your own when the evidence is clean; STOP and ask the user
   (with evidence + a recommendation) when a decision needs intent you cannot get
   from code or tests, or something looks wrong. **Mandatory stops:**
   (a) **Wave B routing** — characterization-test every route encoding first; if
   they don't all produce identical output today, STOP and ask whether the
   divergence is a bug or intended before unifying. (b) any change that would
   alter what's logged to MLflow / written to GCS (experiment names, route
   prefixes, artifact paths/tags) beyond what the plan intends. (c) **Wave B
   binding/destructive operations** — the approved rule is that config-backed
   `Session.active_experiment_id` is the normal experiment source, CLI overrides
   enter only at the session boundary, project-scoped reads may bind without an
   experiment, and destructive cleanup must use explicit targets rather than
   inferring what to delete. STOP only if implementation evidence contradicts
   that rule. (d) a finding that looks based on a wrong assumption, a plan step
   that contradicts the code, or a test failing in a way the plan didn't predict.
   (e) any proposed cleanup that adds a new public noun/CLI verb, registry,
   schema, workflow entry point, or semantic rename not explicitly requested or
   already designed. Treat that as a design decision, not implementation detail.
6. **On stop/pause/blocker:** write a `STATUS.md` Handoff-log entry with the last
   completed step, anything half-done (exact file/step), and the single next
   action, leaving the tree committed or clearly noted.

### Consistency and intent guardrails

The Wave C plan review exposed the main failure mode for the remaining waves:
doing extra "cleanup" that is internally plausible but not requested, not
designed, or inconsistent with the existing codebase. Persist these rules:

1. **Consistency before abstraction.** Before moving code, naming a new module, or
   adding a helper, cross-check the nearest existing domain pattern and follow it.
   Example: project-scoped artifacts belong under `automl/mlflow/project/*`;
   trial artifacts under `automl/mlflow/trial/artifacts/*`; experiment-scoped
   durable objects under `automl/mlflow/experiment/*`.
2. **No new public surface by accident.** Do not add new CLI nouns, commands,
   config knobs, registries, hook entry points, or user-facing names unless the
   plan has explicit user approval for that surface. If a cleanup appears to need
   one, stop and ask.
3. **Typed schemas follow ownership.** JSON artifact schemas live with the domain
   value object that owns the concept; MLflow/GCS seam modules are transport
   writers/readers. If an existing artifact is free-form, first characterize its
   current shape. Only type it when the typed object can round-trip that shape
   exactly; otherwise stop.
4. **No forward-only compatibility shims by default.** When moving an internal
   helper to its correct domain home, do not preserve stale module homes,
   re-export shims, or compatibility wrappers unless the user explicitly
   approves that compatibility surface. Domain ownership beats stale spec text.
5. **Skills are workflow tooling, not core API drivers.** Skill-local scripts and
   hook commands may call the CLI/library, but they should not force new core
   library abstractions or public commands. Keep, defer, or clean them only when
   current skill docs/tests prove the intent.
6. **Plan gates must call out misalignment risk.** At each future wave gate,
   summarize risks that are about intent and design alignment, not code
   difficulty: unrequested surface area, misleading names, inconsistent placement,
   artifact/log shape drift, or behavior that tests cannot adjudicate.

## What we're doing (high level)

The fresh four-layer `automl/` package passed its build but a 2026-05-29
architecture review found ~102 maintainability issues. We are paying them down,
**behavior-preserving**, in **5 sequenced waves**, each a review checkpoint. The
goal is "architecturally merge-ready," not new features.

- **Full task plan:** [`cleanup-plan.md`](cleanup-plan.md) — clusters, the
  finding→wave traceability, per-wave acceptance gates, and Wave A in full
  bite-sized detail. Waves B–E get their detailed steps authored **just-in-time
  at wave start** (that's also where we de-risk — see the plan).
- **Live progress:** [`STATUS.md`](STATUS.md) — current wave, what's done, what's
  next, blockers, and wave-gate commits. **This is the source of truth for
  "where are we."**

## The five waves (ordering is load-bearing)

| Wave | Theme | Risk |
|---|---|---|
| **A** | Hygiene + code-side naming (incl. single `TrialStatus`) | low |
| **B** | Single-source routing + Session→MLflow bind seam (+ `StorageError` fix) | **med-high** |
| **C** | MLflow seam adherence + split the two 900-line monoliths | med |
| **D** | CLI discipline & correctness + validation uniformity | med |
| **E** | Docs/notebook truth + durable test tiers | low |

Hard order: naming(A) → CLI(D) & docs(E); route+bind(B) → monolith splits(C);
docs(E) last. Out of scope (deferred): multi-runner/agent (`docs/to-do/`),
logging wiring, `write_overview`, the 168-site error rewrite. See the plan's
"Out of scope" section.

## Status protocol (how we stay resumable)

The whole point of this folder is that **any session can be stopped and any fresh
session can pick up cleanly.** So:

1. **At wave start / plan gate:** update `STATUS.md` once with the current wave,
   plan path, and approval state.
2. **Within a wave:** do not update `STATUS.md` per task. Keep moving through the
   approved task plan, committing each task with targeted file adds.
3. **On wrap-up (user says stop, or you pause mid-wave):** write a **Handoff log**
   entry at the top of `STATUS.md` with: the last completed step, anything
   half-done (exact file/step), and the single next action. Leave the working
   tree in a committed or clearly-noted state. Never leave a task silently
   half-edited without a handoff note.
4. **On blocker:** mark the wave `BLOCKED`, state why, and what decision is
   needed from the user.
5. **At a wave acceptance gate:** update `STATUS.md` to `WAVE <X> COMPLETE`,
   summarize the task commits and verification evidence, then author the next
   wave's just-in-time plan and return to the plan gate.

`STATUS.md` is progress truth; `cleanup-plan.md` is task truth; the code is
behavior truth. If they disagree, fix the doc.
