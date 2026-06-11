---
name: validate
description: Run AutoML project validation — config structure plus live GCS/MLflow connectivity — before triggering trials.
disable-model-invocation: true
---

# Validate

Use after `/brigit-automl:setup`, before
`/brigit-automl:automl experiment run --project <project_name>`, and any time
`projects/<project_name>/config.py`, `.env` service values,
`projects/<project_name>/PROJECT_INSTRUCTIONS.md`,
`projects/<project_name>/data/pipeline.py` (optional subclass), or
`projects/<project_name>/eval/metrics.py` (optional custom metrics) changes.

From the repo root:

```bash
uv run automl --project <project_name> validate project
```

From inside `projects/<project_name>/...`, `uv run automl validate project`
uses the current project automatically.

One command runs both tiers and prints a JSON `ValidationReport` (non-zero
exit on errors):

- **Structural** (offline, instant)
  - `project.config.*` — `config.py` defines `TASK`, `DATA`, `EVAL`, `RUN_CONFIG`
  - `project.env.*` — `GCS_BUCKET`, `GCS_PREFIX`, `MLFLOW_TRACKING_URI` set in
    `.env` or the environment
  - `project.placeholders` — no scaffold `TBD_` values left in `config.py` or
    Snowflake SQL files
- **Connectivity** (live, a few seconds)
  - `project.connections.gcs` — write/read/delete probe under
    `gs://<bucket>/<prefix>/.validate/`
  - `project.connections.mlflow` — one cheap tracking-server query (auth from
    `MLFLOW_TRACKING_USERNAME`/`PASSWORD` in the environment)
  - `project.connections.snowflake` — live probe, emitted only for
    Snowflake-backed projects: missing `SNOWFLAKE_*` env vars → error listing
    exactly which; else `SELECT 1` (driver errors verbatim) and both SQL files
    must exist on disk. The `SELECT 1` is skipped (warning issued) when
    `RUN_CONFIG.skip_snowflake_live_check = True` or `--no-probe-snowflake` is
    passed; env-var and SQL-file checks still run either way.

After running, render a short per-service report: one line per area
(config, env, placeholders, GCS, MLflow, Snowflake) with pass/fail and the
verbatim issue message on failure. A check ID absent from `issues` means it
passed; Snowflake is "skipped (not a SnowflakeSource project)" when the
project doesn't use it.

If issues are reported, surface them verbatim (do not auto-fix project files).
The check IDs point at the failing surface; consult the matching reference doc:

- `project.config.*` / `project.placeholders` -> [RUN_CONFIG](../../references/setup/run-config.md)
- `project.env.*` / `project.connections.gcs` -> [gcs](../../references/setup/gcs.md)
- `project.connections.mlflow` -> [mlflow](../../references/setup/mlflow.md)
- `project.connections.snowflake` -> [snowflake](../../references/setup/snowflake.md)
- `contracts.data_*` -> [data-pipeline](../../references/setup/data-pipeline.md)
- `contracts.eval_*` -> [evaluation-metric](../../references/setup/evaluation-metric.md)
- `model.*` -> [model-contract](../../references/setup/model-contract.md)

For a deeper end-to-end sanity check (data pipeline, trial runner, MLflow
logging):

```text
/brigit-automl:automl experiment run --project <project_name> --dry-run --max-iter 1
```

Tenets: idempotent, surface errors verbatim, point to docs instead of
auto-fixing project files.
