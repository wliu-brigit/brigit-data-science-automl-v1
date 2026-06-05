# Handoff — continue here

The running note of where the last working session left off. **Read this first
when resuming**, then [`README.md`](README.md) for the docs lifecycle. Keep it
to *current state + next actions* (git history is the changelog, not this).

**Last updated:** 2026-06-05

## Where things stand

- **Active effort: the fraud_anomaly_detection pilot**, on branch
  `feature/fraud-anomaly-detection` (cut from main after the library
  shakedown fixes merged). The project is fully set up and committed; six
  dry-run trials ran during setup (Isolation Forest / PCA / GMM — on
  *different* data snapshots, so their scores are not comparable to each
  other).
- **Background you need before touching it:**
  - Step one of a two-step plan: unsupervised anomaly scoring now (fit never
    sees labels — hard constraint in PROJECT_INSTRUCTIONS); a supervised
    classifier later, once reviewer-confirmed labels exist.
  - The label is a **proxy**: `is_fraud = heuristic_fraud_band ==
    'EXTREMELY_LIKELY'`, a threshold on a heuristic computed upstream from
    the same feature table — so AP measures agreement with the heuristic,
    not ground truth, and high-scoring LOW-band rows (the "discovery queue"
    in the band report) are candidates, not just false positives. The
    non-circular check is the early-default (DPD45) capture metric.
  - Data: Snowflake, wraps the pre-built `fraud_advance_feature_base`
    (never written by the harness; harness owns `..._automl`). Training pull
    is a fixed 99/1 composition, ~1.88M rows, `is_fraud` ≈ 0.17%
    (production is ~0.03% — metric values rank trials, they are not
    deployment claims). 80/20 user-grouped split on SPLIT_PCT.
  - Eval: `AveragePrecision` primary + project-owned metrics in
    `projects/fraud_anomaly_detection/eval/metrics.py` (review-depth
    precision/recall, band report, early-default capture), tests in the
    project's `tests/`.

## Next actions (fraud, in order)

1. **Fair three-way comparison**: `uv run automl --project
   fraud_anomaly_detection --dry-run experiment run --max-budget-usd 1
   --auto-confirm --refresh-data --max-iter 3`. `--refresh-data` is
   required (the eval_pct cleanup changed the training-SQL hash → new
   dataset identity). This puts IF/PCA/GMM on one snapshot with the full
   metric table for the first honest leaderboard.
2. **Bump `per_trial_seconds`** (config) before full scale — eval took 146s
   on 93k rows; 600s will not survive the 1.88M-row dataset.
3. **First full run** (drop `--dry-run`).
4. After a winner: **production-projected precision** (reweight LOW ×~526,
   the known sampling rate) for threshold/capacity setting — industry par is
   ~10–20% alert precision; calibrate expectations accordingly.
5. **Fraud-dive session** on the discovery queue (high-score LOW-band rows
   from the predictions artifact) — generalize findings into explicit
   rules/features/label categories.
6. Reviewer labels → replace the proxy → step-2 supervised classifier.

## Other pending (not fraud)

- **Archive the Snowflake effort** `execution/snowflake-source-and-split-keys/
  → archive/`: its last tail-end item (first real fraud materialize)
  completed 2026-06-05. Follow the effort's own plans-README protocol.
- Library to-dos live in [`to-do/`](to-do/), each named by its ask —
  notably `leaderboard-dataset-pinning.md` (the loop compared AP across
  snapshots twice during the pilot) and `loop-observability.md` (the run is
  silent while it works).
- `main` is local-only ahead of `origin/main` (the shakedown merge has not
  been pushed).

## On hold — waiting, not next fixes

- **MLflow server upgrade** — waiting on the platform team:
  [`to-do/upgrade-mlflow-server.md`](to-do/upgrade-mlflow-server.md).
- **Agent observability follow-ups** — live-run checklist in
  [`to-do/agent-observability-follow-ups.md`](to-do/agent-observability-follow-ups.md) §0.
- **Forward work:** [`to-do/agent-orchestration/`](to-do/agent-orchestration/).
- **Parked:** untangle the MLflow-seam import cycles (extract shared
  value-types into a low contracts layer).

## Gotchas (don't relitigate)

- Repo-local plugins don't flag-free auto-load in Claude Code (v2.1.159). Load via
  **`--plugin-dir agent-skills`** (the loop does this) or symlink `agent-skills` into
  `~/.claude/skills/`. The `agents` custom-path manifest field isn't honored — agents
  must live at the plugin's default `agents/`.
- **The Jupyter kernel must point at THIS clone's `.venv`.** VSCode can launch a
  sibling clone's venv even when you pick "Brigit AutoML (.venv)" — confirm with
  `import sys; print(sys.executable)` in cell 1 (`from automl import experiment`
  failing means the kernel is bound to the wrong clone).
