---
name: automl-guide
description: Use when the user asks how the Brigit AutoML plugin works, what something does, where code lives, or how concepts fit together. Read-only orientation plus pointers for digging into the real code.
---

# Brigit AutoML — guide

Use this skill when the user is **trying to understand** the system — asking
"how does X work?", "where is Y?", "what's the difference between A and B?",
or "should I do this or that?". This skill is read-only. It does not run
setup, validation, trials, or any operational command — point the user at
the matching operational skill if they actually want to *do* something.

## How to answer questions

When the user asks a question:

1. Start from the orientation below — it covers the big-shape stuff most
   questions land on.
2. If the question goes deeper than the orientation, **read the actual code**
   (Read, Grep, Glob). Don't guess from memory; the codebase is the source
   of truth. Quote file paths and line numbers in your answer so the user
   can follow.
3. If multiple files are relevant, name them all rather than picking one
   arbitrarily. The user can decide where to look first.
4. Anchor every answer to a concrete file. If you find yourself saying "I
   think it works like this," go read it.

## Orientation — what AutoML is

AutoML is two things glued together:

- **A Python library** (`automl/`) — the deterministic parts: data sources,
  evaluation framework, MLflow store, the runner that executes one trial,
  CLI verbs.
- **A set of Claude Code skills + sub-agents** (`agent-skills/`) — the
  judgment parts: the agent loop that alternates a proposer turn (which
  reads MLflow and decides what to try next) and a coder turn (which writes
  `model.py` for one trial and runs it).

Both layers operate against one **working project** at
`projects/<project_name>/`. Each project owns two required files: `config.py`
(defines `TASK`, `DATA`, `EVAL`, and `RUN_CONFIG`) and `PROJECT_INSTRUCTIONS.md`.
The agent never touches anything outside its current trial directory.

## How a run actually works

```
automl --project <name> experiment run   (the operational entry point — see /brigit-automl:automl)
    │
    └─ spawns one `claude` subprocess with the /brigit-automl:automl skill
        │
        └─ each iteration the main session:
            ├─ reads MLflow context (trial history, learnings, leaderboard)
            ├─ asks the automl-proposer agent → Proposal JSON
            ├─ validates the proposal
            ├─ creates a trial directory (copies a seed model.py)
            ├─ asks the automl-coder agent → edits model.py, runs trial
            └─ trial logs to MLflow, hooks publish timeline artifacts
```

MLflow is the durable state of the loop. Local files under
`.cache/automl/tmp/` are only for session coordination (locks); they aren't
trusted across sessions.

## Where things live

When the user asks "where is X?", these are the right files to point at:

| Concept | File / directory |
|---|---|
| Loop entry point (Python side) | `automl/agent/launch.py` + `automl/cli/experiment.py` |
| Loop protocol (agent side, prose) | `agent-skills/skills/automl/SKILL.md` + `agent-skills/references/loop/protocol.md` |
| Proposer agent definition | `agent-skills/agents/automl-proposer.md` |
| Coder agent definition | `agent-skills/agents/automl-coder.md` |
| Project entry point types | `automl/project/task.py`, `automl/data/spec.py`, `automl/eval/`, `automl/project/run_config.py` |
| Data contract / sources | `automl/data/` (pipeline.py + `automl/data/sources/`) |
| Model base class | `automl/model/base.py` |
| Evaluation contract | `automl/eval/` |
| MLflow store + artifact routing | `automl/mlflow/` |
| Per-trial execution | `automl/runner/` |
| Trial lifecycle (create / fork / promote) | `automl/trial/` and `automl/cli/trial.py` |
| Session lock | `automl/cli/trial.py` (`trial lock`) |
| Hooks (subagent lifecycle) | `agent-skills/hooks/hooks.json` + `agent-skills/hooks/agent_timeline.py` |
| Skill bodies | `agent-skills/skills/<name>/SKILL.md` |
| Skill-local glue | `agent-skills/skills/<name>/scripts/` |

The discoverable CLI surface is `automl --help` (point users at this when
they want to know what commands exist). Every operational skill is a thin
wrapper around one or two CLI verbs.

## Concepts worth knowing

- **The loop is LLM-driven, not a state machine.** The `claude` binary is
  spawned once; the iteration is then controlled by the agent reading
  SKILL.md prose and MLflow context. There is no Python counter enforcing
  `max_iterations`. See `docs/to-do/agent-orchestration/archived/loop-state-machine.md` — this is a
  known limitation, with the history of how the original state-machine
  attempt got removed.
- **MLflow is durable state, GCS is heavy bytes.** Trial records, metrics,
  leaderboards, learnings — MLflow. Parquet datasets, validation
  fixtures, raw hook events — GCS. The split keeps the tracking server
  fast and artifact retrieval cheap.
- **`cloudpickle` is the model output contract.** Every model serializes to
  cloudpickle and ships through MLflow PyFunc. This is safe specifically
  because deployment uses one shared Docker image (same Python version,
  same pinned packages). Don't propose alternative serialization formats.
- **Dry-run is not proposal-only.** A dry-run uses the dry-run data
  volume, dry-run MLflow route, and dry-run GCS prefix, but it still
  creates trial directories, dispatches the coder, runs `run.py`, and
  produces real MLflow trial runs. The skill SKILL.md guards against the
  common misreading that dry-run means "skip execution."
- **Skill-local `scripts/` are plumbing, not logic.** Anything reusable
  lives in `automl/` (the library). Skill-local scripts are short shims
  that import from `automl/` and produce the JSON context packet for that
  skill's SKILL.md to embed. See `CLAUDE.md` for the
  three-tier convention.

## What this skill does NOT do

Operational tasks belong in the matching skill:

- "Set me up" → `/brigit-automl:setup`
- "Validate my project" → `/brigit-automl:validate`
- "Profile the data" → `/brigit-automl:profile`
- "Run the loop" → `/brigit-automl:automl experiment run --project <name>`
- "Show me the leaderboard" → `/brigit-automl:inspect`
- "Run one steered iteration" → `/brigit-automl:automl experiment run --project <name> --max-iter 1 --instruction "..."`
- "Author or promote a trial by hand" → notebooks 3/4 or `automl trial create --training-origin human` / `trial promote`

If the user wants to *do* one of those, route them to that skill instead of
walking them through it from here.

## When prose isn't enough

For depth on specific loop topics, read the matching `agent-skills/references/loop/`
file directly when the question warrants it: `protocol.md` (turn
structure + stop conditions), `leakage.md` (coder read/write boundaries),
`mlflow-context.md` (the context-JSON schema), `timeouts.md` (per-trial
SIGALRM policy).

For engineering invariants (where new code goes, why the loop is
LLM-driven, the three-tier convention), point the user at
`CLAUDE.md`.
