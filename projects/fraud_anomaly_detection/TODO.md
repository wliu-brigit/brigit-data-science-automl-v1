# To-do — fraud_anomaly_detection

Parked items to revisit. Not status, not learnings (see `LEARNINGS.md` for
those) — things we've decided are worth doing and don't want to lose.

## Feature engineering (parked 2026-06-05)

The round-2 finding (the heuristic's ring signal is ~one feature family)
means model quality is currently feature-limited. The general nugget behind
the circular pattern: **sharing of a scarce resource across many fresh
identities** — bank account today; the same pattern mined from other
columns below. We own the upstream SQL and can recreate/extend the base
table as needed.

- **Device/IP graph — EVIDENCE-BACKED, next in line (screened 2026-06-06 on
  `v1_42baf0ba`).** Within-snapshot screening of `device_id` sharing:
  **≥3 users on one device → 88.3% never-paid gross (n=471), and 69 rows the
  scenario register can't see at all validating at 81.6% never-paid** — the
  first block-tier-grade capture on a new axis (`ring_device_sharing`
  scenario candidate; same no-innocent-version story as the identity burst:
  households don't share 3 Brigit accounts on one phone). Counter-finding:
  raw IP sharing alone is worthless (≥3 users → 1.0x, carrier NAT /
  households); signup-IP sharing only ~3x. Blocked on upstream work only:
  the screening count is whole-snapshot (not as-of-time, and LOW
  downsampling makes it a floor) and the YAML register needs a real as-of
  column — add `users_on_device_id_72h/7d` to the base table exactly like
  the `users_on_bank_account_*` family, rebuild, re-screen as-of, then
  register the scenario as a one-bullet trigger.
- **Speed of monetization** — time from signup → bank link → first advance;
  whether the first action maxes the available amount. Rings move fast;
  real users meander. (Timestamps largely present in metadata.)
- **Bank-account quality** — account age at link time, name match between
  bank holder and identity, deposit history depth (payroll present vs empty
  shell account).
- **ACH return reason codes** (if reachable) — R10/R05 unauthorized =
  fraud-shaped vs R01 insufficient funds = credit-shaped. Also prime
  **label** material, sharper than DPD45.
- **Graph features** — connected-component size/growth over shared
  device + IP + bank + address; generalizes the ring signal beyond one
  bank account (the existing `network_*` columns are aliases of the
  bank-account count, not a real graph).

## Other parked

- **Withhold experiment** — refit anomaly models excluding the
  heuristic-component family (`users_on_bank_account_*` + aliases) to see
  whether the rest of the feature space independently finds the same
  frauds. Shelved 2026-06-05: not ready to set those features aside while
  they're genuinely indicative.
