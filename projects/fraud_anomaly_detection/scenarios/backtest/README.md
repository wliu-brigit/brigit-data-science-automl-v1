# scenarios/backtest — month-over-month scenario backtest

Standalone, warehouse-facing tooling that runs the codified fraud scenarios
([`../register.yaml`](../register.yaml)) against the **full** warehouse history,
**bucketed by advance month**, to watch caught-volume and outcome quality over
time. This is the **"backtest by month"** step of the promotion gate in
[`../../SCENARIOS.md`](../../SCENARIOS.md) — a scenario stays `draft` until it
holds up here.

It runs **outside** the AutoML harness (same status as the upstream feature
DDL): kept in the repo so the analysis is reproducible, executed by hand
(needs VPN). Design notes + the optimization story are in
[`results/OPTIMIZATION_LOG.md`](results/OPTIMIZATION_LOG.md); the original spec
is `docs/execution_parallel/month-over-month-backtest/` (archives over time).

## Files

| file | what it does |
|---|---|
| `monthly_backtest.py` | the backtest. Builds one combined SQL from the params + `SCENARIOS` at the top, runs it, prints + saves a CSV. |
| `heuristic_band_by_month.sql` | the production `heuristic_fraud_band` over the **pre-built snapshot** table, same metrics by band — the rule-of-thumb baseline to compare scenarios against. |
| `heuristic_band.py` | runs that `.sql` in-session and saves to `results/heuristic/`. |
| `profile.py` | targeted query profiling (`GET_QUERY_OPERATOR_STATS` / `run_and_profile` / `explain`) — stop guessing where the time goes. |
| `results/` | output CSVs (gitignored) + `OPTIMIZATION_LOG.md` (tracked). |

## Run

```bash
# scenario backtest — set TEST_WINDOW at the top of the module:
#   ("2025-12-01","2025-12-02")  one-day smoke test (fast)
#   None                          full OUTPUT_START..OUTPUT_END sweep (~11 min)
uv run python -m projects.fraud_anomaly_detection.scenarios.backtest.monthly_backtest

# heuristic-band comparison over the snapshot (seconds):
uv run python -m projects.fraud_anomaly_detection.scenarios.backtest.heuristic_band
```

Both write CSVs named by window; the heuristic output uses the **same column
schema** as the scenario output (a `band` is reported in the `scenario` column),
so the two stack for direct comparison.

## Output columns (one row per month × scenario)

`advance_month · scenario · n_advances · n_scenario · scenario_rate ·
total_loan_disbursed · scenario_loan_disbursed · n_matured · n_dpd45 ·
dpd45_rate · baseline_dpd45_rate · scenario_never_paid_rate ·
baseline_never_paid_rate · scenario_never_paid_principal ·
baseline_never_paid_principal`

Outcome definitions (validated on the snapshot — see the log):
- **`dpd45_rate`** = `n_dpd45 / n_matured` (matured only) — the bust-out cut.
- **`never_paid_rate`** = `never_paid / (repaid + never_paid)` where
  `never_paid = matured AND DPD45 AND not repaid`; lower-is-better like
  `dpd45_rate`; still-open advances excluded. **Charge-off is NOT the loss leg**
  (it is set on only 399 of 10.7M snapshot rows).
- **`never_paid_principal`** = `$` `loan_amount` disbursed to never-paid
  advances (gross principal; ~86% is the realized loss after ~14% recovery).

## ⚠️ Adding or changing a scenario — TWO places must stay in sync

`register.yaml` is canonical, but the SQL predicates here are **hand-written
mirrors** (the engine compiles the register to pandas, not SQL). When you add
or change a scenario, do the register steps **and** update this tool:

1. **Register** (`register.yaml`): append/edit the entry, bump `version`, pin
   the predicate in `tests/test_scenarios.py`, refresh `validation.py` evidence.
2. **Backtest** (`monthly_backtest.py`): add/edit the matching `(name, SQL
   predicate)` in the `SCENARIOS` list and bump `REGISTER_VERSION` to match.
3. Run the monthly backtest — that result is the promotion-gate stat.

If you skip step 2, the new scenario is silently absent from the backtest.
