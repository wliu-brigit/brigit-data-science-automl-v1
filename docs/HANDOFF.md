# Handoff — continue here

Temporary hand-over note: where the last session left off and what's open, so a
new session can pick up. **Not a changelog** — keep only what's current and
relevant; detail lives in the project docs. Rewritten at each wrap, not appended
to.

**Last updated:** 2026-06-09 (graph effort closed out + v3 dataset live → start here)

## How to pick up (per wendao)

Don't dive into code. (1) Read this plus the project docs below; (2) summarize
where things stand; (3) **recommend 2–3 options and let wendao pick.** The next
move is wendao's call, not a queue to drain.

Project docs (`projects/fraud_anomaly_detection/`): **`TODO.md`** (start with the
top **▶ v3 IS LIVE — START HERE** section); **`LEARNINGS.md`** (newest first —
the 2026-06-09 graph entry, then the 2026-06-08 v2/edge entries); `SCENARIOS.md`;
`scenarios/register.yaml`.

## The big picture (what we know, distilled)

The project's settled shape: **anomaly scores are discovery-only; precise
conjunctive scenarios are the product** (SCENARIOS.md). After many rounds:

- **Block-tier (≥90%) precision lives ONLY in the sharing-edge scenarios already
  locked** (`ring_device_burst`, `ring_shared_persistent_account`,
  `ring_account_reuse`, `ring_identity_burst`). These are the product core.
- **The residual (heuristic- and scenario-missed) is exhausted for block-tier**
  — confirmed SEVEN independent ways (greedy/fine tree, beam search, unsupervised
  component, seed-proximity ×2, structural). What remains there is **review-tier
  (~5–7×)**: the neobank × fresh-account × small-amount fast-churn cohort.
- **Graph / multi-hop entity rings (the 2026-06-08/09 effort, on the OLD v1 data)
  — explored and closed with a clear verdict:** multi-hop *does* find genuinely
  net-new fraud, but it is **review-tier and vanishingly low-coverage (~0.01% of
  transactions)**, not the ≥90%-with-volume we hoped for. Specifics worth
  carrying into v3:
  - **What works:** advance-co-occurrence edges (user↔device/bank/persistent),
    degree-capped (~20) to kill shared-infra junk nodes; the discriminator is a
    **small, dense, MULTI-resource-type component** (not big single-type ones).
  - **Best durable rule:** structural `comp≥5 & types≥2` — stable ~55–65%
    out-of-time across ~7 rings, net-new — but ~0.006% coverage.
  - **Best feature:** count of OTHER DPD45-bad users sharing your resources
    (self-excluded; own prior default is credit-history, not ring). Durable
    out-of-time precision ~40% / ~7×, review-tier; best as a model feature or a
    review/clawback queue, NOT a real-time block.
  - **Don't repeat these mistakes:** day-bucketing drops intra-day bursts (use
    timestamp-ordered same-day linking); raw IP is a giant-junk-component
    generator (never a node); small-n early pockets look like 100% but are 1–3
    rings and regress to the true rate — always check distinct-ring count +
    out-of-time before believing a precision number.
  - Reusable tooling (pinned to old v1, re-point at v3): `analysis/
    graph_discovery_sweep.py` (engine + rule battery), `graph_validate_winner.py`
    (concentration + out-of-time), `graph_seed_coverage.py` (coverage/seed
    diagnostic).

## v3 dataset is LIVE (use this now)

Built 2026-06-09 (the planned rebuild, executed). **Use `use_project(...,
dry_run=False)`** and dataset id **`v2_2ac98b52`** (the `v2_` prefix is the
lineage counter; `schema_version` is still 1 — the known naming quirk).

- Table `fraud_advance_feature_base_automl_v3`, **2,412,045 rows × 115 cols**.
- **Date span 2025-01-01 → 2026-06-08** (1.54M rows in 2025, 0.87M in 2026) —
  the deeper history that fixes the old Dec-1 left-censoring.
- **New graph-node keys (the headline add):** `email_key` (100% filled),
  `phone_key` (100%), `address_key` (97.9%; rest sentinel-screened → NULL) — all
  SHA-256 hex, ~1.38M distinct each. These are hashed surrogates, not raw PII.
- Sharing edges populated with deeper history: bank 4,897 rows ≥2 users, device
  5,466, persistent 478, phone 2,291, address 2,649, email 593 (**email max=6 —
  still near-unique/noise, consistent with prior findings; treat email as a weak
  node at best**).
- `name_match_official` and `official_name` both pruned (the noise columns). 115
  cols = v2's 112 + the 3 new keys.
- Split health (the recurring gotcha) green: train 1.93M / 2,573 pos (0.133%),
  test 482k / 664 pos (0.138%); 3,237 `is_fraud` total at natural prevalence.

## Training/eval pooling decision (wendao, 2026-06-09 — apply on v3)

Keep the full Jan-2025→now span in the BASE table (it's the as-of history that
feeds the graph/edges), but **pool the training/eval rows from `training_data.sql`
to: `feature_as_of_ts >= 2025-08-01` AND mature (`label_mature_d45 = 1`).**

- **Why Aug-2025 onward:** gives the entity graph + as-of windows a ~7-month
  warm-up before the first scored row, so rings aren't left-censored (the v1
  Dec-1 problem).
- **Why mature-only:** every train/test row then has its full 45-day DPD45 label
  resolved (no undefined labels diluting precision — the exact issue raised in
  review). In practice this also trims the most-recent ~45 days (not yet mature).
- Note the naming: `training_data.sql` pulls the rows for BOTH train and
  test/validation splits (confusing name; SPLIT_PCT then divides them). The
  filter belongs there.

## What's open — wendao to pick (focused next steps on v3)

1. **Re-baseline on v3** with the Aug-2025 / mature-only pool: confirm the locked
   scenarios fire (the register requires the edge columns and raises on a missing
   one), re-measure their precision/volume on the deeper history, and establish
   the new residual. Foundation for everything else.
2. **★ Graph with the NEW node types + deeper history (the highest-value test).**
   Re-point `graph_discovery_sweep.py` at `v2_2ac98b52` and add `phone_key` /
   `address_key` as node types (email is noise — skip or weak). The multi-type
   density discriminator is *what worked*; more node types (types could reach 5)
   + deeper history is the direct lever on the v1 ceilings — does coverage and/or
   precision move off review-tier/0.01%? This decides whether the graph is worth
   more than a feature.
3. **Bad-neighbour-count as a model feature.** With Aug-2025 training start, far
   more seeds have matured before the first scored row, so the feature is better
   populated than on v1. Test it in the supervised lens / as a feature (it's
   review-tier as a standalone rule).
4. **If the graph still caps at review-tier/low-coverage on v3** → the lever for
   the OTHER residual cohort (neobank fast-churn, which the graph can't touch) is
   Tier-3 data: **income/payroll** (shell-vs-real discriminator), IP-intelligence
   (datacenter/VPN/geo), ACH return codes. Parked in TODO Tier-3; this is the
   pivot if graph-on-v3 disappoints.

## Gotchas (still live)

- **Use `dry_run=False` and `v2_2ac98b52`.** The register requires the edge/key
  columns and raises on a missing column → run `validation` with `--no-dry-run`.
- **`config.exclude_cols` edits only take effect on the next materialize** (the
  registry is baked at materialize time).
- `experiment run` / `data materialize` preflight needs **Snowflake/VPN** (~30s).
- `--instruction` is **shell-evaluated** by the loop's context-render — keep it
  plain prose (no `->`, parens).
- A killed `experiment run` holds the session lock (~6h self-expiry); release via
  `trial lock release` (ids in `.cache/automl/tmp/session_locks/*.lock/`).
- `base_table.sql` comments must avoid `;` and `'` until the `_scrub_sql` harness
  bug is fixed (TODO library follow-ups).

## State / loose ends

- **v3 SQL + materialization committed** (`2db51bd`, the parallel session). Graph
  analysis suite + findings committed (`5ea6d3d`, `f93a786`). This wrap-up commit
  prunes the 4 superseded intermediate graph scripts (findings preserved in
  LEARNINGS; recoverable from git) and updates the docs.
- **Register NOT edited** — no graph scenario registered (mid-comparison register
  edits break comparability; the graph verdict is review-tier so any new rule is
  a review-tier candidate, wendao's call). Recommended candidate if pursued:
  `ring_multitype_structural` (`comp≥5 & types≥2`) as a review-tier scenario.
- Retained analysis tooling: graph suite (`graph_discovery_sweep`,
  `graph_validate_winner`, `graph_seed_coverage`) + the prior screens
  (`feature_due_diligence`, `unsupervised_lens`, `edge_precision_screen`,
  `residual_next_layer`, `subgroup_discovery`, `ip_screen`, `institution_screen`,
  `ceiling_probe`, `supervised_lens`).
