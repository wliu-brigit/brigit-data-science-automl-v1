# Archived query SQL — v2 (inlined upstream + Tier-1 features)

Frozen 2026-06-09, superseded by the **v3** rebuild now living at
`data/queries/base_table.sql`. Do not edit these files.

## What v2 was

The first **single-step** base table: the previously out-of-band upstream
(`fraud_advance_feature_base`) was folded directly into `base_table.sql`, so the
harness builds everything in one materialize step. The same rebuild added the
**Tier-1 feature set** (see the project `TODO.md` "CONSOLIDATED FEATURE-ADD
PLAN"):

- as-of scarce-resource sharing edges — device, persistent-id, address, phone,
  email (`users_on_*_{72h,7d,30d}`, mirroring `users_on_bank_account_*`);
- Jaro-Winkler name-match (`name_match_first` / `name_match_last`);
- `is_neobank_high_risk_institution`, `is_joint`;
- prior-only advance velocity (`prior_min_hours_between_advances_on_account`);
- identity→advance speed in hours;
- the team's bank-sharing detection flags (3-in-72h, 5-ever, 10-ever) — all as-of.

`heuristic_fraud_score` / `heuristic_fraud_band` were kept **byte-identical** to
v1 (proxy-label comparability).

## What it materialized to

- **Table:** `fraud_advance_feature_base_automl_v2`
- **Dataset id:** `v1_76d3ad45` (1,021,950 rows × 113 cols)
- **Window:** anchors from `2025-12-01` (one month), history from `2025-11-01`.

Note: the materialized `v1_76d3ad45` still carries the `name_match_official`
noise column and the `official_name` dead-carry — both removed in v3.

## Why v3 replaced it

- **History depth** extended: anchors `2025-12-01` → `2025-01-01`, history
  `2025-11-01` → `2024-01-01` (the left-censoring fix — windowed/cumulative
  features need real prior depth).
- **Graph-node keys emitted** for the joined-but-dropped entities (email, phone,
  address — SHA-256 hashes of their normalized + sentinel-screened values) so
  multi-hop ring detection can use them with no raw PII emitted.
- `official_name` pruned (dead carry — it was the account product type, not the
  holder name; its only consumer `name_match_official` was already dropped).
