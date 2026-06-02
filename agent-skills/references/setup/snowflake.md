# Snowflake setup

## What is this

Credentials for Snowflake, used when the project's data lives in a Snowflake
warehouse.

## Why we need it

Only projects whose `config.py` `DATA.source` uses `SnowflakeSource` talk to Snowflake. The
pipeline runs the two SQL queries under `projects/<project_name>/data/queries/`
to materialize train + test data, then writes them to GCS.

If your project uses `LocalCSVSource` or `GCSParquetSource` instead,
**skip this doc** — Snowflake credentials are not part of the run path.

## How to set it up

Ask your data engineering team for the values below, then fill them in `.env`:

```
SNOWFLAKE_ACCOUNT       # e.g. ab12345.us-east-1.aws
SNOWFLAKE_USER          # your username
SNOWFLAKE_PASSWORD      # your password or auth token
SNOWFLAKE_WAREHOUSE     # the warehouse your role can use
SNOWFLAKE_ROLE          # the role you query under
SNOWFLAKE_DATABASE      # database the tables live in
SNOWFLAKE_SCHEMA        # schema within that database
```

There's no `gcloud auth`-style one-time setup — Snowflake auth is per-request
using these values.

## Common gotchas

- **Network access.** Corporate VPN or cloud sandboxes may need data
  engineering to whitelist your IP or VPC before connections succeed.
- **Role permissions.** Your role needs `SELECT` on the specific tables your
  SQL queries reference. Permission errors surface as Snowflake driver
  errors — surface them verbatim; the user resolves with their data team.
- **Connector version.** Very old Snowflake connector versions can fail to
  connect to newer accounts. The runbook pins recent versions; this is only
  an issue if a project pins an older one.
