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

- **Device/IP graph** — same `device_id` / `ip_address` across users (the
  device/IP analog of users-on-bank-account); datacenter/VPN IP detection;
  signup IP vs disbursement IP distance. *Both columns are already in the
  snapshot as metadata — derivable without new upstream data.*
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
