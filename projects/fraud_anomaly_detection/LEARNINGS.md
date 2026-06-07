# Learnings — fraud_anomaly_detection

High-level takeaways only: what worked, what didn't, what surprised us.
Append as they emerge; date each entry. (Long-term these belong in MLflow at
the experiment/project level — this file is the ad-hoc home until the
workflow settles.)

## 2026-06-06/07 — scenario register built; the proxy label retired

**The mechanism (now the project's shape)**
- Scenarios codified as a declarative YAML register
  (`scenarios/register.yaml`) compiled and run by a register-agnostic
  engine: edits never rebuild the dataset, predicates are unit-tested, and
  the conditions are data — mechanically compilable to SQL at ship time.
- Matched rows are out of the model's world on both sides: dropped from fit
  (gate, a hard instruction the opus coder followed first try) and masked
  out of every model metric (`ResidualOnly`); they surface only as rule
  outcomes (`scenario_identified`). Verified arithmetically against the
  logged baseline trial — every band count, positive, and denominator.
- Scenarios match **independently**; per-scenario gross + unique capture
  bound the contribution without order-dependence. Sequencing is an action
  policy (highest tier wins), never a measurement policy.

**The register absorbing the heuristic (the milestone)**
- `ring_account_reuse` (ex-S1b): the 7d window beat lifetime — the 33
  lifetime-only matches were **0/17 never-paid** (stale account reuse is
  innocent); 96.4% never-paid at n=251. 7d ≡ 72h: reuse is ring-cycling
  speed.
- `ring_identity_burst` (ex-S1): ≥3 identities *created within 72h* on one
  bank account; 89.2% never-paid, unique-vs-sibling 78.7%. Pure advance
  velocity without fresh identities collapsed to 23–28% on unique capture
  (one user re-advancing is normal product use) — deliberately not
  registered.
- Together: E_L band 100% covered, union 89.1% never-paid at 16x base —
  so the proxy label has no positives left in the residual and the primary
  moved to **never-paid AP on mature residual rows**.
- Honest discovery stat (captured LOW-band rows): **2** — both never-paid.
  Today's register *replaces* the heuristic; it barely out-discovers it yet.

**The next axis (screened, recorded in TODO.md)**
- Device sharing ≥3 users: 88.3% gross, **69 register-invisible rows at
  81.6% never-paid** — block-tier-grade on a genuinely new axis; blocked
  only on as-of `users_on_device_*` base-table features. Counter-finding:
  raw IP sharing alone is worthless (~1x; carrier NAT/households).

**Process**
- The iteration loop works as designed: question → variant table against
  the pinned snapshot → register edit → version bump → evidence refresh →
  tests. No SQL, no rebuild, no model run needed to refine a rule.
- The loop's preflight validates the Snowflake connection even for pinned
  no-refresh runs (VPN required for ~30s of preflight; nothing after).

## 2026-06-05 — round 2 (pinned snapshot `v1_42baf0ba`, 98/2, opus/high)

**The leaderboard (first honest one — all trials on one snapshot)**
- IF 0.995 · kNN 0.577 · GMM 0.415 test AP. The ordering is the textbook
  prediction for an axis-aligned extreme-tail target in a redundant noisy
  feature space — harness, split, and eval all behaved.

**The central finding: the proxy label is circular by construction**
- `is_fraud` (band = EXTREMELY_LIKELY) is a *deterministic function of six
  model input features*; reconstruction matches 100%. One column
  (`users_on_bank_account_7d`) alone ranks AP 0.998. Verified not leakage —
  the IF fit code is clean; the label is just written in the model's own
  feature space.
- Consequence: **AP-vs-proxy is saturated and uninformative** for model
  comparison. DPD45-at-depth is mostly circular too (the E_L band runs
  82.8% DPD45). The honest metric is **within-LOW lift** on never-paid
  DPD45 (`label_gross_dpd45=1 AND label_repaid_current_snapshot=0`).

**The factor model of the current feature space**
- The heuristic = ring/identity-sharing (~80 of 100 pts; `network_*` are
  SQL aliases of the same count) + account-level advance velocity (~20 pts;
  effectively *is* the POSSIBLE band).
- **Create-to-withdraw speed is not in the heuristic and is the strongest
  blind-spot factor** (2.4–2.9x within LOW). E_L median identity→advance =
  12 *minutes* vs ~15 months for everyone else.
- Missing device/IP telemetry: ~2x, tiny n. Amount is a *modifier*:
  protective alone (0.5x — limits are earned), 8–14x inside newness.
- All blind-spot factors collapse to ~one latent dimension (newness/speed);
  a genuinely third factor needs the TODO.md feature engineering.

**The stance shift (the big one)**
- **Anomaly models are discovery-only; scenarios are the product.** Named
  conjunctive scenarios with a behavioral theory + disqualifiers, tiered by
  validated precision — not additive point scores. Framework, register, and
  industry grounding: `SCENARIOS.md`. First refinement loop immediately
  found S1b (fresh identity + account with prior advance history = ring
  seen from the account side, 89.5% never-paid).
- Reusable discovery tooling: `analysis/rule_discovery.py` (attribution →
  residual queue → surrogate rules → enrichment, all from one MLflow run id).

**Process**
- Pinning the snapshot before the run (materialize first, no
  `--refresh-data` on the run) worked exactly as intended.
- sonnet/medium coder failed the baseline on a preprocessor/column-mask
  ordering bug; opus/high ran 3/3 clean on the retry.
- A killed `experiment run` leaves the session lock held (~6h self-expiry);
  release needs the session+lock ids from
  `.cache/automl/tmp/session_locks/*.lock/metadata.json`.

## 2026-06-05 — pilot round 1 (deleted; setup shakedown)

Six trials; only one (GMM on `v5_91f5af2a`, ~93k rows) ran on a sound setup.
The whole round was archived — these are the takeaways, not results to build on.

**Signal**
- Unsupervised scoring works here: GMM test AP 0.139 at 0.12% prevalence
  (~110x lift over random). Reference point, not a leaderboard.
- The non-circular check passed: early-default (DPD45) precision at top-0.5%
  depth was 29% vs 5.8% base (~5x lift) — the score carries real risk signal
  beyond heuristic agreement.
- The discovery queue is populated: ~40% of the top-0.5% rows were LOW-band.
- PCA reconstruction error: not a good fit for this problem — dropped.

**Process**
- Pin one snapshot per experiment before comparing anything: 5 of 6 trials
  were wasted on failed or degenerate-sample setups (different snapshots,
  splits with 175–8k rows). Sanity-check row/positive counts per split right
  after materialization, before spending trials.
- Per-trial overhead is large at dry-run scale: ~120s constant pyfunc
  logging + ~30s data load vs 34s of actual fit. Budget accordingly.
- With ~23 test positives (99/1 pull at 100k rows) small AP deltas are
  noise. Round 2 doubles the labeled share (98/2) for ~2x positives per
  split; still worth establishing a noise floor (re-run one config twice on
  the same snapshot) before trusting model-vs-model deltas.
