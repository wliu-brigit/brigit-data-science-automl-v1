# brigit-automl

brigit-automl is an **AutoML harness** — a standardized workflow for building
tabular ML models faster. You describe a prediction problem once as a project
recipe, and the harness runs a repeatable loop that proposes, implements, and
evaluates model trials, logging every trial to MLflow and writing artifacts to
GCS so results stay reproducible and comparable. Today it runs as a Claude Code
skill; the core is a plain Python library you can also use directly.

## Install

```bash
git clone <repo-url> brigit-automl
cd brigit-automl
uv sync
```

Python >=3.11 is required. All work goes through `uv` (e.g. `uv run automl ...`).

Shared tooling lives in the default `dev` dependency group, synced
automatically. Packages that only one project under `projects/` needs live in
a dependency group named after that project — opt in with
`uv sync --group <project>`; each project's README has the details.

## Prerequisites

Before running on real data:

| Service | What you need |
| --- | --- |
| **MLflow** | A tracking server reachable at `MLFLOW_TRACKING_URI` — durable state lives here. |
| **GCS** | Application-default credentials for the bucket that holds datasets and artifacts: `gcloud auth application-default login`. |
| **`.env`** | Copy `.env.example` to `.env` and fill `GCP_PROJECT`, `GCS_BUCKET`, `GCS_PREFIX`, and `MLFLOW_TRACKING_URI`. |

## Getting started

`projects/example_homecredit/` is a complete, runnable example — the best way
to learn the workflow hands-on. Start with the **notebooks** in
[`projects/example_homecredit/notebooks/`](projects/example_homecredit/notebooks/);
they walk the whole lifecycle and are made to be run and tinkered with:

| Notebook | What you'll do |
| --- | --- |
| `0_understand_project_sessions_and_routes` | The core concepts — projects, sessions, routing. |
| `1_define_and_materialize_dataset` | Define a dataset and materialize it to GCS. |
| `2_profile_logged_dataset` | Profile a logged dataset (EDA). |
| `3.1_run_agent_automl` | Run the agent-driven AutoML loop. |
| `3.2_author_new_trial` | Author a model trial by hand. |
| `3.3_fork_existing_trial` | Fork and tweak an existing trial. |
| `4_reevaluate_existing_model` | Re-evaluate a model on new data. |
| `5_inspect_logged_runs_and_artifacts` | Inspect logged runs and artifacts. |

Run them with the project's venv as the kernel — register it once **from the repo
root** so the kernel binds to *this* clone's `.venv`:

```bash
uv run python -m ipykernel install --user --name brigit-automl-venv --display-name "Brigit AutoML (.venv)"
```

The notebooks are pinned to that kernel (`brigit-automl-venv`), so `import automl`
resolves to this repo's `.venv`. (In VS Code, select the `.venv` interpreter
instead.) If you have several brigit-automl clones, re-run the command from the
one you want, then **restart the notebook kernel** — a kernel left pointing at
another clone (or a system `python3`) imports the wrong `automl` and fails on
`from automl import experiment`.

Prefer the command line? A single dry-run iteration runs end-to-end against an
isolated MLflow experiment and GCS prefix:

```bash
uv run automl --project example_homecredit --dry-run experiment run -- --max-iter 1
```

For library use, start with `automl.use_project(...)`, then import the domain
modules you need:

```python
from automl import data, experiment, trial, eval
```

## Claude Code skills

The interactive skills (`/brigit-automl:setup`, `:automl`, `:validate`,
`:profile`, `:inspect`, `:propose`, `:coder`, `:automl-guide`) live in the
`agent-skills/` plugin. Three ways to load them:

1. **The AutoML loop — nothing to do.** `automl experiment run` (and the
   `/brigit-automl:automl` skill) launch Claude with the plugin loaded
   automatically; running the loop needs no setup.
2. **One session, with a flag** — from the repo root:
   ```bash
   claude --plugin-dir agent-skills
   ```
   Then `/brigit-automl:setup` and friends are available — live, nothing installed.
3. **Every session, auto-loaded** (recommended for interactive use) — symlink
   the plugin into your user skills directory once. From the repo root (safe to
   re-run; works whether or not it's already set up):
   ```bash
   mkdir -p ~/.claude/skills && ln -sfn "$PWD/agent-skills" ~/.claude/skills/brigit-automl
   ```
   The skills then load in **every** Claude session, **live** from your current
   checkout (it's a symlink — it follows whatever branch you have checked out,
   no cache, nothing to update). To remove it:
   ```bash
   rm ~/.claude/skills/brigit-automl
   ```

The symlink is user-global (skills appear in all your sessions, pointing at this
clone). Claude Code has no flag-free auto-load for a plugin checked into a repo,
so interactive use needs either the symlink or the `--plugin-dir` flag.

## Design principles

- **A standardized, reproducible workflow.** One recipe per problem; every
  trial is logged and comparable, not a throwaway script.
- **MLflow is the durable record; GCS holds the heavy bytes** (datasets,
  artifacts). Nothing else is a source of truth.
- **One model packaging contract** (`cloudpickle`) on one shared serving image.
- **`uv` only** — never `pip` or the system Python.

## For contributors

See [`CLAUDE.md`](CLAUDE.md) for the package layout, conventions, and the
invariants to preserve.
