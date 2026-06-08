# Month-over-month scenario backtest

**Status:** execution (design approved 2026-06-07, implementation pending)
**Owner:** wendao
**Lineage:** the "Month-over-month backtest" candidate in `docs/HANDOFF.md`;
the promotion gate the two draft block-tier scenarios need
(`projects/fraud_anomaly_detection/scenarios/register.yaml`).

## The ask

The codified scenarios were validated on a single pinned dry-run snapshot
(107K rows, one output month). That answers "do these predicates separate
fraud *here*" but not "how does the catch hold up over time." This effort
runs the **same scenario predicates** against the full warehouse history,
**bucketed by advance month from Jan 2025 to now**, so we can watch the
caught-volume and the dpd45 quality trend month over month — and normalize
for changing advance volume with a denominator.

This is a backtest / ops tool, **not** harness-wired modelling. It runs
outside the AutoML loop, the same status as the upstream feature DDL
(`data/queries/upstream_fraud_advance_feature_base.sql`): kept in the repo so
the analysis is reproducible, executed by hand against the warehouse.

## Why a stripped, combined query

The upstream feature SQL is ~750 lines and took ~1hr to build one month at
~10M rows, because it computes the full feature set (KYC, device/IP, network
score, heuristic bands, marketing attribution, 24h/30d/90d/lifetime windows,
prior-loan-amount aggregates). **The scenario predicates need almost none of
it** — only two derived velocity features. Stripping to just those is the
whole point: the backtest stays cheap enough to run across ~18 months.

Original instinct was one SQL per scenario. Revised (2026-06-07) to **one
combined query** emitting **long format — one row per (month × scenario)** —
so the shared scaffolding (anchor advances, bank-account mapping, the
velocity CTEs) lives in one place and a change propagates to every scenario.
We still compute only the features the registered scenarios actually read, so
the cost saving stands.

## The two scenarios, as SQL

Mirroring `register.yaml` v2026-06-06.3 (both `block` / `draft`):

| scenario | match expression |
|---|---|
| `ring_account_reuse` | `DATEDIFF('hour', identity_created_time, feature_as_of_ts) <= 24 AND loan_amount > 100 AND prior_advances_on_bank_account_7d > 0` |
| `ring_identity_burst` | `users_on_bank_account_72h >= 3` |

Plus a synthetic `scenario_any` row per month (the union — headline "total
caught"). The match expressions are hand-written SQL that mirror the register
(the engine compiles the register to pandas, not SQL); they are NOT
auto-generated from the YAML. When the register changes, these are updated by
hand and the register version in the file header is bumped.

## What survives the strip

Kept CTEs:
- `all_advances` — `advance_id`, `user_id`, plaid routing/account,
  `loan_amount`, `feature_as_of_ts` (= `origination_timestamp`, the true
  disbursement time and the month bucket), and **only the dpd45 outcome
  fields**: `label_gross_dpd45`, `label_mature_d45`,
  `label_repaid_current_snapshot`. Windowed from `history_start`.
- `bank_account_links` — user → `bank_account_key`, `identity_created_time`
  (the identities/plaid tables are not advance-windowed, so burst counts stay
  correct across the whole backtest).
- `anchor_advance_account_candidates` — anchor advances (output window) mapped
  to bank account, **deduped to one row per `advance_id`** (replicating the
  upstream's final `QUALIFY ROW_NUMBER() … PARTITION BY advance_id`) so the
  denominator never double-counts a fan-out.
- `users_on_bank_account_72h` — distinct users whose identity was created in
  the 72h before the advance, on the same bank account.
- `prior_advances_on_bank_account_7d` — distinct prior advances on the same
  bank account within 7d.

Dropped: KYC/Socure, client metadata/device/IP, network score, heuristic
score + bands, marketing attribution, the 24h/30d/90d/lifetime windows, all
`prior_loan_amount_*` aggregates.

## Output schema (one row per month × scenario)

| column | meaning |
|---|---|
| `advance_month` | `DATE_TRUNC('month', feature_as_of_ts)` |
| `scenario` | `ring_account_reuse` / `ring_identity_burst` / `scenario_any` |
| `n_advances` | denominator — all anchor advances that month |
| `n_matched` | advances the scenario flagged |
| `match_rate` | `n_matched / n_advances` (volume-normalized) |
| `sum_loan_amount_matched` | matched dollar exposure that month |
| `n_mature` | matched advances with `label_mature_d45 = 1` |
| `n_dpd45` | matched & mature & `label_gross_dpd45 = 1` |
| `dpd45_rate` | `n_dpd45 / n_mature` — **the key metric** |
| `baseline_dpd45_rate` | dpd45 rate over *all* mature advances that month |

`dpd45_rate` denominator is `n_mature` (matches `validation.py` — rate among
matured matches), deliberately not `n_matched`; recent immature months show a
small/zero `n_mature`, which keeps them honestly flagged rather than
silently diluted.

## Parametrization

A Python wrapper assembles the SQL from an f-string with params at the top,
for readability and cheap testing:

```python
OUTPUT_START      = "2025-01-01"    # first anchor month
OUTPUT_END        = "2026-06-01"    # exclusive upper bound (≈ now)
HISTORY_BUFFER    = "1 month"       # lookback before OUTPUT_START for velocity
TEST_SINGLE_MONTH = "2025-12-01"    # a month → restrict the window to it; None → full run
EXECUTE           = True            # True → fetch_df returns a DataFrame; False → print SQL only
```

`TEST_SINGLE_MONTH` is the cheap-validation lever: validate on one month,
then set `None` for the full multi-month run. Velocity windows are ≤7d, so a
1-month history buffer before `OUTPUT_START` is plenty.

## Where it lives & how it runs

```
projects/fraud_anomaly_detection/scenarios/backtest/
├── __init__.py
└── monthly_backtest.py
```

Run: `uv run python -m projects.fraud_anomaly_detection.scenarios.backtest.monthly_backtest`

Execution reuses the existing warehouse seam
`automl.utils.io.snowflake.fetch_df(sql)` (`.env` creds, Arrow fetch, decimal
coercion). When `EXECUTE` is False it prints the rendered SQL for pasting into
the Snowflake console. Running against the warehouse needs VPN (known
gotcha).

## Out of scope

- Auto-generating SQL from the YAML register (predicates differ enough
  scenario-to-scenario that hand-written SQL mirroring the register is the
  pragmatic call for now).
- Wiring the backtest into the harness / MLflow.
- Case-sample review and sign-off (separate promotion-gate step).
