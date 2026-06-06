# Scenario-based fraud detection — framework & register

The detection stance for this project: **named fraud scenarios with a
behavioral theory, conjunctive triggers, and explicit disqualifiers** — not
an additive point score. Anomaly models are **discovery-only**: they surface
candidate patterns offline; nothing ships on a model score. A score's only
legitimate job is ranking *within* a scenario's queue.

Why not additive points: weights are arbitrary ("why is this 25?"), and
addition ignores dependencies — in this data, high amount is *protective*
alone (0.5x) and 8–14x dangerous inside a newness conjunction. Conditions
mean something only inside the right story.

An entity may match multiple scenarios; overlap is itself a signal, not
double-counting.

## The rubric — every scenario defines six fields

| field | requirement |
|---|---|
| **Theory** | Why a *fraudster* behaves this way. The story must be about intent — never "legit user who can't pay back". |
| **Trigger** | Conjunction of conditions (ALL hold). Each condition necessary to the story. Composable as a SQL WHERE / pandas filter. |
| **Disqualifiers** | Release conditions — evidence the legit explanation is more likely (the credit-risk firewall). |
| **Supporting indicators** | Raise confidence / rank the queue; never gate membership. |
| **Validation** | never-paid-DPD45 (`label_gross_dpd45=1 AND label_repaid_current_snapshot=0`) lift now → sampled case review → reviewer labels. |
| **Volume** | Projected alerts at production scale (apply the LOW-band reweight). |

**Action tiers by validated precision** (never-paid-DPD45 on mature rows):

- **Block-tier**: ≥ ~80% — decline/hold, near-certain loss.
- **Mitigate-tier**: ~10–50% — cap amount, step-up verification; never block.
- **Review-tier**: ~2x lift+ — human queue / shadow tag only.
- **Discovery**: model residual queues — analysis input, no user-facing action.

## Scenario register

All numbers from dry-run snapshot `v1_42baf0ba` (107k rows, 2026-06-05);
thresholds are provisional until the ATL/BTL sweep on the full pull.

### S1 — Ring bust-out via identity sharing — BLOCK-tier (validated)

- **Theory:** one operator cashes out through a mule bank account using many
  freshly created identities.
- **Trigger:** ≥3–4 identities on the bank account within 72h AND multiple
  advances through the account (the current heuristic's E_L band).
- **Supporting:** identity-creation burst rate, machine-speed monetization
  (E_L median identity→advance = 12 minutes).
- **Validation:** 83% never-paid at band level; 98.9% with speed+amount
  confirmation. n=58 mature (test).

### S1b — Ring cash-out via account reuse — BLOCK-tier candidate

- **Theory:** same ring story, observed from the account side: the *identity*
  is hours old but the *bank account* already has advance history — a
  day-old user cannot have taken those prior advances, so other identities
  drew through this account before.
- **Trigger:** identity→advance ≤ 24h AND amount > $100 AND
  `prior_advances_on_bank_account_lifetime` > 0.
- **Validation:** **89.5% never-paid, n=237 mature** — independent
  re-expression of S1 through different columns (no identity-count join;
  real-time friendly). Includes ~15 LOW-band rows the heuristic misses.
- **Status:** needs monthly backtest + case sample before promotion.

### S2 — Solo fast monetization — MITIGATE-tier

- **Theory:** a stolen/synthetic identity monetized quickly — but
  indistinguishable at trigger time from a real person in a same-day cash
  crunch. No proven intent → no block.
- **Trigger:** identity→advance ≤ 24h AND first-ever advance on the account
  (`prior_advances_on_bank_account_lifetime` = 0).
- **Disqualifiers (by construction + future):** a ≤24h identity cannot have
  repaid history, so the user-level release condition is auto-satisfied;
  add payroll-deposit presence / bank-account tenure when those features
  exist (TODO.md).
- **Treatment:** cap advance amount / step-up verification.
- **Validation:** 13.1% never-paid (2.4x), n=306 mature with amount>$100;
  ~13% across variants. Case review needed to estimate the fraud fraction.

### S3 — Telemetry evasion — REVIEW-tier

- **Theory:** emulators/automation suppress device fingerprinting; a real
  phone can't easily produce no device ID. Only meaningful with newness —
  alone it's also just old app versions.
- **Trigger:** missing device_id+ip AND fresh account.
- **Validation:** ~2x, n=197 (tiny volume — cheap shadow tag).

### S4 — Device/IP ring (future)

S1's story told through device/IP sharing instead of bank-account sharing.
Blocked on the device/IP graph features in [`TODO.md`](TODO.md).

## Governance (industry pattern)

1. **Backtest by month** on the full pull — stability, volume, precision.
2. **Threshold setting via ATL/BTL** (above/below-the-line): sample
   just-under-threshold cases to verify the cut, don't eyeball.
3. **Shadow mode** before action: tag in the warehouse ~90 days, watch
   precision/volume against live traffic.
4. **Champion/challenger** for changes: one change, contained population,
   review before promotion.
5. **Case sampling** (15–20/scenario) — doubles as the reviewer-label seed
   for the step-2 supervised model.

## Industry grounding (sources retrieved 2026-06-05)

- **FFIEC BSA/AML Examination Manual** — regulator-governed scenario/typology
  monitoring; the origin of the scenario-library structure.
- **FinCEN advisories** — published typologies as the unit of detection.
- **Unit21 / Sardine engineering docs** — vendor practice: scenario rule
  validation via offline backtest + shadow mode + precision/alert-volume
  gates before deployment.
- **Protiviti / Abrigo whitepapers** — ATL/BTL threshold tuning and periodic
  review methodology.
- **FICO** — champion/challenger strategy governance.
- **Experian** — first-payment-default / never-pay as the outcome proxy that
  separates first-party fraud from credit risk.
- Academic fraud-rule-mining surveys — conjunctive rules over additive
  scores; the feature-dependency failure mode of weighted sums.
