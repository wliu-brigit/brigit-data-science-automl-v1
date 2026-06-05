---
name: setup
description: Pick a working project, wire up the services AutoML needs, and validate.
disable-model-invocation: true
---

# Setup

Guides you from a fresh clone to a project that's ready to run trials. Three
things happen here:

1. Pick the project you'll be working in (or create a new one).
2. Wire up service endpoints in `.env` and project metadata in `config.py`.
3. Validate the setup before running real trials.

The agent never enters credentials. When a service doc asks for a value, you
open `.env` in your editor and fill it yourself.

## Step 1 — Pick or create the working project

A "project" is one working directory under `projects/<project_name>/`. It owns
the data contract, evaluation contract, config, and project direction. Every
`automl` command operates against exactly one project at a time.

List the projects that already exist:

```bash
ls projects/
```

For each project folder, peek at one line to summarize it:

```bash
for p in projects/*/; do
  name=$(basename "$p")
  proj="$p/config.py"
  if [ -f "$proj" ]; then
    target=$(grep -oE "BinaryClassification\(target=['\"][^'\"]+['\"]" "$proj" 2>/dev/null | head -1 | sed -E "s/.*target=['\"]([^'\"]+)['\"].*/\1/")
    source=$(grep -oE "[A-Za-z0-9_]+Source\(" "$proj" 2>/dev/null | head -1 | sed -E 's/\(//')
    echo "  $name - target=$target, source=$source"
  fi
done
```

Then ask:

> Which project would you like to set up for? Pick one of the above, or say
> "new" and I'll create a fresh Snowflake-backed project for you.

If the user picks **new**, ask for a short snake_case name (e.g.
`fraud_scoring`). Then create the project through the CLI; `/setup` is the
single entry point and the CLI owns the file creation:

```bash
uv run automl --project-root . project init <new_name>
```

Do not ask which template to use. The internal default is Snowflake, and the
generated project intentionally keeps user-owned values as `<TBD>`.

Tell the user: *"`projects/<new_name>/` is ready. Edit
`projects/<new_name>/config.py` directly; it is intentionally heavily
commented and lists the common source, metric, split, and custom-pipeline
options inline. The recipe has `<TBD>` slots to fill in, including
`experiment_id`, `TASK.target`, `DATA.source` fields, and the Snowflake SQL
placeholders — we'll get to those in Step 3."*

Disposable / dev projects: use the `dev_` prefix (e.g. `dev_my_dataset`).
Folders starting with `dev_` are gitignored by default, which makes them the
standard place for local dry-run experiments. `example_homecredit` is the
committed example and testing ground, so its recipe, sample data, helper
script, and notebooks ship with the repo.

## Step 2 — Wire up services

For the picked project, three services may be needed:

| Service | When | Where credentials go |
|---|---|---|
| **GCS** | Always | `.env` (`GCP_PROJECT`, `GCS_BUCKET`, `GCS_PREFIX`) |
| **MLflow** | Always | `.env` (`MLFLOW_TRACKING_URI`, username/password) |
| **Snowflake** | Only if `projects/<name>/config.py` uses `SnowflakeSource` | `.env` |

For each service the project needs, read the matching reference doc and
present it to the user one at a time:

- `${CLAUDE_SKILL_DIR}/../../references/setup/gcs.md`
- `${CLAUDE_SKILL_DIR}/../../references/setup/mlflow.md`
- `${CLAUDE_SKILL_DIR}/../../references/setup/snowflake.md` (only if SnowflakeSource)

After each doc, prompt the user:

> Open `.env` in your editor and fill the values listed above. Say "next" when
> you're done.

To check which source the project uses:

```bash
grep "Source(" projects/<project_name>/config.py
```

Skip Snowflake if the source is `LocalCSVSource` or `GCSParquetSource`.

## Step 3 — Walk the project recipe

Each project owns two required files. Give the user one short pointer to each,
then let them ask for deeper detail on whichever they want:

```text
projects/<project_name>/ has two files you'll edit:

1. config.py            - the active, heavily commented Python recipe. Edit it
                           directly. It defines four constants: TASK (target
                           column + task type), DATA (DataSpec with source and
                           column roles), EVAL (EvalSpec with primary metric),
                           and RUN_CONFIG (experiment_id, models,
                           per_trial_seconds).
                           SQL-backed projects also edit data/queries/base_table.sql
                           and data/queries/training_data.sql.
                           References:
                           agent-skills/references/setup/run-config.md  (RUN_CONFIG fields)
                           agent-skills/references/setup/data-pipeline.md  (DATA / DataSpec)
                           agent-skills/references/setup/evaluation-metric.md  (EVAL / EvalSpec)

2. PROJECT_INSTRUCTIONS.md - domain notes the agent reads every turn:
                             constraints, things to try or avoid, open
                             questions. Reference:
                             agent-skills/references/setup/project-instructions.md
```

If the user asks for a specific reference, read it and present it, then return
to this menu.

## Step 4 — Validate setup

Once the env file and recipe files are filled in, hand off to the validate
skill — tell the user: *"Let's validate the setup."* — then follow
`/brigit-automl:validate`. It runs one command covering both structure and
live connectivity:

```bash
uv run automl --project <project_name> validate project
```

That checks (check IDs in parentheses):
- `config.py` defines `TASK`, `DATA`, `EVAL`, and `RUN_CONFIG` (`project.config.*`)
- `.env` or the process environment defines `GCS_BUCKET`, `GCS_PREFIX`, and
  `MLFLOW_TRACKING_URI` (`project.env.*`)
- No scaffold `TBD_` placeholders remain in `config.py` or the Snowflake SQL
  files (`project.placeholders`)
- GCS is reachable with a write/read/delete probe under the project prefix
  (`project.connections.gcs`)
- The MLflow tracking server answers an authenticated query
  (`project.connections.mlflow`)
- Snowflake-backed projects get a live probe: missing `SNOWFLAKE_*` env vars
  are an error listing exactly which; otherwise `SELECT 1` runs (driver errors
  verbatim) and both SQL files must exist (`project.connections.snowflake`)

End with a short per-service report (config / env / placeholders / GCS /
MLflow / Snowflake — pass, fail, or skipped), quoting failure messages
verbatim. Point the user at the matching reference doc by check ID, as listed
in `/brigit-automl:validate`.

Do not auto-fix the user's project files. The user fixes; we re-run.

## Step 5 — Hand-off

When validation passes, end with:

```text
Setup is good for `<project_name>`. One last sanity check is the dry-run:

uv run automl --project <project_name> --dry-run experiment run --max-budget-usd 1

That prepares the dataset, creates a trial, runs it, and logs to the
dry-run MLflow route. If it succeeds, you're ready for a real run:

uv run automl --project <project_name> experiment run
```

## Tenets

- Agent never enters credentials. If the user pastes one in chat, refuse and
  ask them to put it in `.env` directly.
- Read at most one or two reference docs per turn.
- Surface service errors verbatim — don't paraphrase or guess at fixes.
- GCS and MLflow are required; Snowflake is conditional on the active
  project's source.
- Validation in Step 4 covers structure and live GCS/MLflow connectivity; the
  Step 5 dry-run remains the deeper end-to-end check (data pipeline, trial
  runner, MLflow logging).
