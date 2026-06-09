# Archived base-table query SQL — version history

Frozen snapshots of every retired base-table definition, one folder per
version. The **active** SQL always lives at `data/queries/base_table.sql`
(config.py `base_table_sql` points there); when it is rebuilt, the prior version
is snapshotted here before editing. These files are reference only — do not edit.

The version number tracks the **table name** suffix
(`fraud_advance_feature_base_automl` → `_automl_v2` → `_automl_v3`), NOT the
dataset id. Dataset ids read `v1_<hash>` because the `v1` is the recipe-schema
version (fixed) and the hash changes with the SQL — a known naming quirk.

| Version | Folder | Approach | Table | Dataset |
|---|---|---|---|---|
| v1 | `v1_two_step_upstream/` | Two-step: out-of-band upstream DDL + thin harness SELECT | `fraud_advance_feature_base_automl` | — |
| v2 | `v2_inlined_tier1/` | Inlined upstream + Tier-1 feature set | `fraud_advance_feature_base_automl_v2` | `v1_76d3ad45` |
| v3 | *(active — `data/queries/base_table.sql`)* | v2 + extended history (Jan-2025 anchors) + graph-node keys | `fraud_advance_feature_base_automl_v3` | *(materializes to a new `v1_<hash>`)* |

Each version folder carries its own README with the full detail. See the
project `TODO.md` "CONSOLIDATED FEATURE-ADD PLAN" and "★ NEXT SQL REBUILD" for
rationale.
