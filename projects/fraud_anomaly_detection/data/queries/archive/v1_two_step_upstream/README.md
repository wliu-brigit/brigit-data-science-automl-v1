# Archived query SQL — the old two-step base-table approach

Superseded 2026-06-08 by the inlined `data/queries/base_table.sql`.

## What the old approach was

The feature base was built in **two steps**:

1. `upstream_fraud_advance_feature_base.sql` — a `CREATE OR REPLACE TABLE`
   DDL, run **out-of-band** (manually, in `SANDBOX_WLIU`), that read the raw
   dbt/production tables and materialized `fraud_advance_feature_base`.
2. `base_table.legacy.sql` — the thin harness SELECT
   (`SELECT * FROM …fraud_advance_feature_base`) that the harness wrapped in
   `CREATE OR REPLACE TABLE {base_table}` and into which it injected `SPLIT_PCT`.

## Why we changed it

- We folded the full upstream logic **into** `base_table.sql`, so the harness
  builds everything in one materialize step (no out-of-band dependency on a
  hand-maintained `fraud_advance_feature_base`).
- The same rebuild added the Tier-1 feature set (see the project `TODO.md`
  "CONSOLIDATED FEATURE-ADD PLAN"): as-of sharing edges (device, persistent-id,
  address, phone, email), Jaro-Winkler name-match, `official_name` holder match,
  `is_joint`, `IS_NEOBANK_HIGH_RISK_INSTITUTION`, prior-only advance velocity,
  identity→advance speed, and the team's bank-sharing detection flags (3-in-72h,
  5-ever, 10-ever) — all as-of.

## Consequence (verify at preflight)

The inlined `base_table.sql` now reads the raw tables directly
(`fct_loans`, `base_prod__*`, `user_client_metadata`, …) instead of the
pre-built `fraud_advance_feature_base`. The harness's Snowflake role must have
read grants on those, or the materialize step fails.

These files are frozen for reference — do not edit them.
