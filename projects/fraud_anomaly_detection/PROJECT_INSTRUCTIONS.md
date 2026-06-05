# Project Instructions - fraud_anomaly_detection

Guidance for the loop. The recipe (target, source, splits, metrics) lives in
config.py — facts there are not repeated here.

## Goal

Rank cash-advance disbursements by fraud risk with an **unsupervised anomaly
score**. Better = fraud concentrated at the top of the ranking.

This is step one of a two-step plan (anomaly detection now; a targeted
supervised classifier later). The label is a heuristic proxy
(`heuristic_fraud_band == 'EXTREMELY_LIKELY'`, computed upstream from this
table's own features), so eval measures agreement with the heuristic, not
ground truth. Don't over-tune to small metric differences.

## Constraints (hard)

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
  candidates** (fraud the heuristic may have missed), not just false
  positives — the proxy-label primary metric penalizes them by construction.

## Approaches to try

- Isolation Forest (baseline first), PCA reconstruction error, GMM negative
  log-likelihood, k-means distance-to-centroid, HDBSCAN via distillation
  (fit clusterer on train, distill to an inductive scorer).
- Robust scaling / log1p for the heavy-tailed count and velocity features.

## Approaches to avoid

- Any supervised or semi-supervised use of the target in fit (see Constraints).
- Transductive methods without a distillation step.

## Open questions

- Which feature families drive useful anomalies — velocity counts, network
  structure, or KYC/device signals?
- Does the score concentrate the LIKELY/POSSIBLE bands too (sanity check that
  it generalizes beyond the EXTREMELY_LIKELY definition)?
- High-score LOW-band cases: investigate, and generalize what's found into
  explicit rules / features / label categories (future dedicated session).
