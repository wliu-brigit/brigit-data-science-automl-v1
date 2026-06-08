# Backtest query optimization log

Autonomous run/profile/improve loop on the one-day smoke window
(`TEST_WINDOW = 2025-12-01 .. 2025-12-02`). Each iteration is profiled with
`GET_QUERY_OPERATOR_STATS`, the top bottleneck is fixed, then the result is
validated against the **baseline numbers** below. A mismatch in matched counts
means the change broke correctness (revert). `n_advances`/baseline-rate may
drift by a couple rows between runs because the source tables are live CDC —
that drift is noted, not treated as a logic failure.

## Baseline correct numbers (one day, 2025-12-01)

| scenario | n_advances | n_matched | sum_loan_matched | n_mature | n_dpd45 | dpd45_rate | baseline_dpd45 |
|---|---|---|---|---|---|---|---|
| ring_account_reuse | ~52587 | 2 | 440.0 | 2 | 0 | 0.0 | ~0.0614 |
| ring_identity_burst | ~52587 | 0 | 0.0 | 0 | 0 | NaN | ~0.0614 |
| scenario_any | ~52587 | 2 | 440.0 | 2 | 0 | 0.0 | ~0.0614 |

Invariants that MUST hold (data-drift-independent): `n_matched`,
`sum_loan_matched`, `n_dpd45` per scenario.

## Iteration timings (one-day window)

| # | wall (s) | change | matched (reuse/burst) | validates? |
|---|---|---|---|---|
| 1 | 582.8 | original: 35d buffer, `lifetime` count, all-history joins | 2 / 0 | baseline |
| 2 | 292.6 | drop `lifetime`+`30d`; push 72h/7d bounds into the joins; 8d buffer | 2 / 0 | ✓ (same as #1) |
| 3 | 602.4 | dedup `plaid_accounts` to current state via a **CTE** before joins | 2 / 0 | ✓ numbers, but REGRESSION (slower) |

**Iteration 3 finding (reverted):** the `plaid_current` CTE is referenced
twice (`relevant_account_keys` + `scoped_plaid`); Snowflake re-inlined the
billion-row `QUALIFY` dedup each time instead of computing it once — one
TableScan hit **8.75B rows**. CTE materialization isn't guaranteed, so the
dedup must be forced into a **temp table** (iteration 4).

| 4 | 591.8 | materialize current-state plaid in a **temp table** (dedup once) | 2 / 0 | ✓ numbers, but REGRESSION (slower) |

**Iteration 4 finding (reverted):** materializing forces a `QUALIFY`
ROW_NUMBER window sort over the **entire ~1B-row** CDC view — more expensive
than iteration 2, which only ever deduped the **scoped** subset (plaid rows on
accounts the advance-window users touch). Any approach that touches all 1B rows
with a window/sort regresses.

## Conclusion

**Iteration 2 (rolling-window + entity scoping, 292.6s) is the winner** and is
what the committed code uses. The dominant remaining cost is two scans of the
~1B-row `base_prod__plaid_accounts` CDC view; that cost is *window-independent*
(the same whether one day or 17 months), so the full multi-month run is not
proportionally slower. Deduping the full view up front (iters 3–4) regresses
because the billion-row window sort costs more than the scoped scans. Further
speedups would need a smaller/pre-deduped current-state plaid source, which is
out of scope here.

| best | 292.6s | drop lifetime/30d, push 72h/7d into joins, scope to touched accounts |

## Deliverable runs (saved CSVs)

Run on the optimized (iteration-2) query.

**1-month — December 2025** (`backtest_test_20251201_20260101.csv`, 325.9s,
1,745,593 advances):

| scenario | n_matched | sum_loan_matched | dpd45_rate | baseline_dpd45 |
|---|---|---|---|---|
| ring_account_reuse | 523 | 129,015 | 91.2% | 6.2% |
| ring_identity_burst | 675 | 130,215 | 96.1% | 6.2% |
| scenario_any | 723 | 140,330 | 91.6% | 6.2% |

Confidence check vs the register (validated on the 107K dry-run sample):
reuse 96.4% / burst 89.2%. Full-month rates land in the same high band
(>90%, ~15× baseline) — predicate translation confirmed faithful; small
differences are full-population-vs-sample + the simplified dedup tiebreak.

**Full Jan 2025 -> now** (`backtest_20250101_20260701.csv`, reran 723.3s on the
final column set — `scenario_*`/`never_paid_rate`/`never_paid_principal`; 18
months x 3 scenarios = 54 rows). Denominator `n_advances` ~1.1M-2.1M/month.
Loss exposure (`scenario_never_paid_principal`): negligible pre-Dec, ~$118-126K
in Dec 2025, ~$217-243K at the Jan-2026 peak.

The headline: **the ring pattern switches on at Dec 2025.**
- **Jan-Nov 2025:** scenarios barely fire (reuse 9-33, burst 2-27 per month)
  and dpd45 of the matched rows is low/noisy (often near baseline) — the
  predicates don't separate fraud in this period.
- **Dec 2025 -> Apr 2026:** step-change. Matches jump (reuse 523 -> 1139 peak
  in Jan-2026; burst 675 -> 1547) and dpd45 of matched rows is **90-98%** vs a
  ~5-6% baseline (~15-18x), sustained through Feb, tapering Mar (~74-81%) and
  Apr (~60-68%).
- **May-Jun 2026:** matches present but `n_mature = 0` -> `dpd45_rate = NaN`
  (advances <45 days old as of the run); Jun is a partial month (~510K). Counts
  only, no outcome yet.

Reading: the mule-account ring bust-out appears to be a fraud wave that began
~Dec 2025; before that these triggers are low-volume noise. Worth a closer look
(the pre-Dec matches at near-baseline dpd45 are likely innocent, not early ring
activity). Caveat: dedup tiebreak is simplified (no lifetime/30d), so fan-out
advances may differ slightly from the pinned-snapshot selection.
