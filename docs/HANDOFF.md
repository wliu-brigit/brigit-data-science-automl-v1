# Handoff — continue here

Where the last working session left off. **Read this first when resuming**,
then [`README.md`](README.md) for the docs lifecycle. Keep it to *current
state + next actions* (git history is the changelog, not this). **Written
only when wrapping a session for handoff (or when asked)** — never updated
mid-session as a status log.

**Last updated:** 2026-06-05

## How to pick this up (protocol, per wendao)

Do **not** dive into work directly. On resume: (1) read this file plus the
project docs below, (2) summarize where things stand, (3) **recommend 2–3
options for what to do next and ask wendao to pick or redirect**. The next
move is a judgment call wendao wants to make, not a queue to drain.

Project docs to read (all in `projects/fraud_anomaly_detection/`):
- `SCENARIOS.md` — the detection stance (scenarios over scores; model =
  discovery-only), the rubric, and the scenario register with validation
  numbers. This is the center of gravity now.
- `LEARNINGS.md` — dated takeaways; the 2026-06-05 round-2 entry explains
  why the project pivoted from model comparison to scenario building.
- `TODO.md` — parked feature-engineering menu + the shelved withhold
  experiment.

## Where things stand

- **Branch `feature/fraud-anomaly-detection`.** Round 1 state was deleted
  (archived, not purged — production MLflow). Round 2 ran clean: pinned
  100k dry-run snapshot `v1_42baf0ba` (98/2 composition, ~0.33% positive),
  opus/high agents, 3/3 trials finished — IF 0.995 / kNN 0.577 / GMM 0.415
  test AP.
- **The IF result was investigated, not celebrated**: the proxy label is a
  deterministic function of the model's own features (circular), so
  AP-vs-proxy is saturated and no longer ranks models. The session pivoted
  to factor decomposition and rule discovery — full story in `LEARNINGS.md`.
- **The product direction is now scenario-based rules** (`SCENARIOS.md`):
  S1 ring (validated block-tier), S1b ring-via-account-reuse (89.5%
  never-paid, promotion gate = monthly backtest + case sample), S2 solo
  fast monetization (mitigate-tier), S3 telemetry evasion (review-tier),
  S4 device/IP ring (needs TODO.md features).
- **Reusable tooling**: `analysis/rule_discovery.py` runs the discovery
  pipeline (attribution → residual queue → surrogate rules → enrichment)
  against any logged trial by MLflow run id, read-only.
- Dry-run only so far — full-scale run deliberately deferred until the
  pattern is settled (explicit call, 2026-06-05).

## Candidate next moves (for the pick-one conversation, not a to-do list)

- **S1b promotion gate**: monthly-stability backtest on the full pull +
  15–20 sampled cases (the case review also seeds reviewer labels for the
  step-2 supervised model).
- **Scenario refinement**: keep tightening triggers/disqualifiers against
  the never-paid-DPD45 bar (wendao explicitly wants to revisit).
- **Feature engineering from `TODO.md`** (device/IP first — metadata
  already in the snapshot) → unlocks S4 and a third independent factor.
- **Fraud-dive on the discovery queues** (GMM's 60 LOW-band rows, residual
  queue) for new patterns.
- Housekeeping options: noise-floor rerun of the IF config; LOW-reweight
  volume projections for the scenario register.

## Other pending (not fraud)

- **Archive the Snowflake effort** `execution/snowflake-source-and-split-keys/
  → archive/`: its last tail-end item (first real fraud materialize)
  completed 2026-06-05. Follow the effort's own plans-README protocol.
- Library to-dos live in [`to-do/`](to-do/), each named by its ask —
  notably `leaderboard-dataset-pinning.md` (mitigated this round by
  materializing before the run; not yet fixed in the library) and
  `loop-observability.md` (the run is silent while it works).
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
- A killed `experiment run` leaves the session lock held (~6h self-expiry);
  release with `trial lock release --session-id ... --lock-id ...` using the
  ids in `.cache/automl/tmp/session_locks/*.lock/metadata.json`.