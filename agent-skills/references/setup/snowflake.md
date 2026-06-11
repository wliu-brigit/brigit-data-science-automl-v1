# Snowflake setup

## What is this

Credentials for Snowflake, used when the project's data lives in a Snowflake
warehouse.

## Why we need it

Only projects whose `config.py` `DATA.source` uses `SnowflakeSource` talk to
Snowflake. On materialize, the source ensures the base table exists
(bootstrapping it from the project's `base_table.sql` SELECT with `SPLIT_PCT`
injected from `split_group_key`; rebuilt only on `--refresh-source`), checks
the split invariant against the actual table, then pulls training rows via
`training_data.sql` — the derived dataset's bytes land in GCS, its record in
MLflow. See `data-pipeline.md` for the full contract.

If your project uses `LocalCSVSource` or `GCSParquetSource` instead,
**skip this doc** — Snowflake credentials are not part of the run path.

## How to set it up

Ask your data engineering team for the values below, then fill them in `.env`:

```
SNOWFLAKE_ACCOUNT       # required — e.g. ab12345.us-east-1.aws
SNOWFLAKE_USER          # required — your username
SNOWFLAKE_PASSWORD      # required — your password or auth token
SNOWFLAKE_WAREHOUSE     # optional — defaults to DATA_SCIENCE_WH
SNOWFLAKE_ROLE          # optional — defaults to DATA_SCIENCE_ROLE
SNOWFLAKE_DATABASE      # database the tables live in ({database} substitution)
SNOWFLAKE_SCHEMA        # schema within that database ({schema} substitution)
```

There's no `gcloud auth`-style one-time setup — Snowflake auth is per-request
using these values.

`automl validate project` probes the connection live for Snowflake-backed
projects (`project.connections.snowflake`): missing required env vars are an
error listing exactly which; otherwise it runs `SELECT 1` and surfaces driver
errors verbatim, and checks both SQL files exist on disk. Set
`RUN_CONFIG.skip_snowflake_live_check = True` to skip only the `SELECT 1` when
off-VPN (env-var and SQL-file checks still run); `--no-probe-snowflake` does the
same per-invocation.

## Common gotchas

- **Network access.** Corporate VPN or cloud sandboxes may need data
  engineering to whitelist your IP or VPC before connections succeed.
- **Role permissions.** Your role needs `SELECT` on the specific tables your
  SQL queries reference. Permission errors surface as Snowflake driver
  errors — surface them verbatim; the user resolves with their data team.
- **Connector version.** Very old Snowflake connector versions can fail to
  connect to newer accounts. The runbook pins recent versions; this is only
  an issue if a project pins an older one.
