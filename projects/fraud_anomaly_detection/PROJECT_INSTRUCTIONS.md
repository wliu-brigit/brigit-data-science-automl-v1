# Project Instructions - fraud_anomaly_detection

Guidance for the loop. The recipe (target, source, splits, metrics) lives in
config.py — facts there are not repeated here.

## Goal

Rank cash-advance disbursements by fraud risk with an **unsupervised anomaly
score**. Better = fraud concentrated at the top of the ranking.

Detection is scenario-first (`scenarios/register.yaml`): codified rules
handle what they explain, and **the model's job is discovery on the
residual** — surfacing patterns no scenario covers yet. The primary metric
scores the residual ranking against the real outcome (never-paid gross
DPD45 on mature rows), not the heuristic proxy: the register absorbs the
heuristic's top band, so agreement with it is no longer informative.
Never-paid still contains innocent credit risk — treat the primary as a
direction signal and don't over-tune to small metric differences.

## Constraints (hard)

- **Every trial applies the scenario gate before fit.** Rows a codified
  scenario matches (`scenarios/register.yaml`) are rule-handled, not
  modeled: call `gate_fit` from
  `projects.fraud_anomaly_detection.scenarios.gate` on the train frame
  before fitting. The model's job is discovery on the
  residual; a trial trained on scenario-matched rows is invalid. (Eval
  mirrors this: all model metrics are residual-masked; matched rows appear
  only in the `scenario_identified` report.)
- **The model must never consume the target (or any label-derived column) in
  fit.** Fit is unsupervised: `fit(X)` on train features only — no `y`, no
  supervised objectives, no label-based feature selection, no threshold tuning
  on labels inside the trial. Labels are for **evaluation only**. A supervised
  classifier here is cheating and invalidates the trial.
- Every model must emit a **continuous anomaly score per row** (higher = more
  anomalous) and be **inductive**: fit on train, score frozen on unseen rows.
  No transductive-only methods (e.g. raw LOF/HDBSCAN labels on the test set).
- Do not read test data during fit.
- `exclude_cols` in config.py (heuristic score/band, repayment/DPD outcome
  fields) are leakage — never re-add them as features.

## Domain notes

- Rows are advance disbursements (Dec 2025+); features are as-of origination
  time (identity/bank-account sharing velocity, prior-advance velocity,
  network score, KYC, device/IP).
- The training pull is downsampled (LOW band sampled, non-LOW kept), so
  metric values are not production-prevalence numbers — compare trials
  against each other, not against deployment expectations.
- Fraud is ring-shaped: shared bank accounts across many fresh identities,
  burst velocity. Expect anomalies to cluster, not be i.i.d. outliers.
- In the band report, LOW-band rows ranked near the top are **discovery
  candidates** (fraud the heuristic may have missed) — under the never-paid
  primary they are where a model can genuinely win, not false positives.

## How to explore

- **Establish a baseline first**: a plain Isolation Forest with simple
  preprocessing, on the experiment's pinned snapshot. Every later trial is
  judged against it on the same snapshot.
- After the baseline, model choice is **open-ended** — propose whatever the
  evidence suggests (other IF variants, GMM negative log-likelihood, k-means
  distance-to-centroid, HDBSCAN via distillation, autoencoders, ...). Vary
  one idea per trial and say in the hypothesis what the trial should teach us.
- Robust scaling / log1p for the heavy-tailed count and velocity features.

## Approaches to avoid

- Any supervised or semi-supervised use of the target in fit (see Constraints).
- Transductive methods without a distillation step.
- PCA reconstruction error — tried in the first pilot round; not a good fit
  for this problem, don't re-propose it.

## Learnings log

High-level takeaways (what worked, what didn't, what surprised us) go in
[`LEARNINGS.md`](LEARNINGS.md) — append as they emerge.

## Open questions

- Which feature families drive useful anomalies — velocity counts, network
  structure, or KYC/device signals?
- Does the score concentrate the LIKELY/POSSIBLE bands too (sanity check that
  it generalizes beyond the EXTREMELY_LIKELY definition)?
- High-score LOW-band cases: investigate, and generalize what's found into
  explicit rules / features / label categories (future dedicated session).
