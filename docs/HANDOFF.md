# Handoff — continue here

Where the last working session left off. **Read this first when resuming**,
then [`README.md`](README.md) for the docs lifecycle. Keep it to *current
state + next actions* (git history is the changelog, not this). **Written
only when wrapping a session for handoff (or when asked)** — never updated
mid-session as a status log.

**Last updated:** 2026-06-07

## How to pick this up (protocol, per wendao)

Do **not** dive into work directly. On resume: (1) read this file plus the
project docs below, (2) summarize where things stand, (3) **recommend 2–3
options for what to do next and ask wendao to pick or redirect**. The next
move is a judgment call wendao wants to make, not a queue to drain.

Project docs to read (all in `projects/fraud_anomaly_detection/`):
- `scenarios/register.yaml` — **the canonical scenario register**: doc 1 =
  definitions (the file users edit), doc 2 = machine-owned validation
  evidence. Refresh evidence with
  `uv run python -m projects.fraud_anomaly_detection.scenarios.validation`.
- `SCENARIOS.md` — the prose stance, rubric, governance, and *uncodified*
  candidates (register is canonical for codified rules).
- `LEARNINGS.md` — dated takeaways; the 2026-06-06/07 entry explains the
  scenario-register build and why the proxy label was retired.
- `TODO.md` — the device/IP find (evidence-backed, next in line) + parked
  feature menu.

## Where things stand (after the 2026-06-06/07 session)

- **Branch `feature/fraud-anomaly-detection`**, all work committed
  (`5872721`). 627 tests green (unit + contracts + project).
- **Scenario-first detection is built and verified end to end.**
  `scenarios/` package: YAML register compiled by a register-agnostic
  engine (declarative conditions → pandas masks; SQL-compilable at ship
  time), fit gate (matched rows never trained on), residual-masked eval
  (matched rows in no model metric; rule outcomes in
  `scenario_identified`), evidence-refresh script. Register edits never
  rebuild the dataset.
- **Two draft block-tier scenarios** (both mule-account bust-out typology,
  register v2026-06-06.3, evidence on pinned dry-run snapshot
  `v1_42baf0ba`):
  - `ring_account_reuse` (ex-S1b): fresh identity ≤24h + amount>$100 +
    advance on the account within 7d. **96.4% never-paid, n=251.**
    (Lifetime→7d tightening: the 33 lifetime-only matches were 0/17
    never-paid — stale reuse is innocent.)
  - `ring_identity_burst` (ex-S1): ≥3 identities *created within 72h* on
    one bank account. **89.2% never-paid, n=400; unique-vs-sibling 78.7%.**
- **Register absorbs the heuristic**: union 421 rows @ 89.1% never-paid
  (16× base 5.5%); E_L band **100% covered**, LIKELY 63%, POSSIBLE ~2%.
  Honest discovery stat (captured LOW rows): **2** (both never-paid) — the
  register replaces the heuristic but barely out-discovers it yet.
- **Primary metric swapped** to `residual_never_paid_average_precision`
  (proxy AP is degenerate on the residual). One gated trial ran clean
  (opus, `4_isolation_forest_baseline`, dry-run experiment
  `fraud_anomaly_v1`); the gate + residual eval were verified
  arithmetically against its logged report.
- Pure-velocity cycling (advances_7d≥3 alone) deliberately NOT registered:
  23–28% never-paid on unique capture — mitigate/review material pending
  case review.

## Candidate next moves (for the pick-one conversation, not a to-do list)

- **Round-3 AutoML iteration on the gated residual** — wendao leans here
  next (no data update needed). The ring is rule-handled and the primary is
  outcome-based, so anomaly trials now answer "what's left that we can't
  explain?" — discovery queues (`analysis/rule_discovery.py` works by run
  id) feed new scenario candidates. Same pinned snapshot, dry-run
  experiment; `--max-iter` 2–3.
- **Device/IP graph features** — evidence-backed in TODO.md (device shared
  ≥3 users: 81.6% never-paid on 69 register-invisible rows) but requires a
  base-table rebuild; **wendao explicitly deferred data updates
  (2026-06-07)** — don't start without a fresh go-ahead.
- **Month-over-month backtest** (full-history Snowflake pull + the same
  register predicates) — the promotion gate both drafts need; also
  re-confirms thresholds (7d window, ≥3/72h) per cohort. Read-only on the
  warehouse but a bigger pull than dry-run.
- **Disqualifiers + sign-off conversation** — both scenarios have empty
  disqualifiers (the rubric's credit-risk firewall) and `status: draft`;
  the case-sample review (15–20/scenario) doubles as reviewer-label seed.

## Other pending (not fraud)

- **Archive the Snowflake effort** `execution/snowflake-source-and-split-keys/
  → archive/` (tail-end item completed 2026-06-05; follow the effort's own
  plans-README protocol).
- Library to-dos in [`to-do/`](to-do/) — notably
  `leaderboard-dataset-pinning.md` and `loop-observability.md`. New
  candidate worth filing: preflight validates the Snowflake connection even
  for pinned no-refresh runs (see Gotchas).
- `main` is local-only ahead of `origin/main` (the shakedown merge has not
  been pushed).

## On hold — waiting, not next fixes

- **MLflow server upgrade** — waiting on the platform team:
  [`to-do/upgrade-mlflow-server.md`](to-do/upgrade-mlflow-server.md).
- **Agent observability follow-ups** —
  [`to-do/agent-observability-follow-ups.md`](to-do/agent-observability-follow-ups.md) §0.
- **Forward work:** [`to-do/agent-orchestration/`](to-do/agent-orchestration/).
- **Parked:** untangle the MLflow-seam import cycles.

## Gotchas (don't relitigate)

- **`experiment run` preflight requires Snowflake reachability (VPN) even
  for pinned no-refresh runs** — the CLI's `validate project` hardcodes
  live connection checks. VPN is needed only for the first ~30s; nothing
  after preflight touches the warehouse. Candidate library to-do, not yet
  filed.
- Old round-2 trials show as **unscored** on the leaderboard — they lack
  the new primary (`residual_never_paid_average_precision`). Expected, not
  a bug; round-3 trials aren't comparable to round 2 by design.
- Editing `scenarios/register.yaml` mid-experiment breaks trial
  comparability — bump `version`, edit between rounds, and rerun the
  evidence refresh.
- Repo-local plugins don't flag-free auto-load in Claude Code (v2.1.159). Load via
  **`--plugin-dir agent-skills`** (the loop does this) or symlink `agent-skills` into
  `~/.claude/skills/`.
- **The Jupyter kernel must point at THIS clone's `.venv`** — confirm with
  `import sys; print(sys.executable)` in cell 1.
- A killed `experiment run` leaves the session lock held (~6h self-expiry);
  release with `trial lock release --session-id ... --lock-id ...` using the
  ids in `.cache/automl/tmp/session_locks/*.lock/metadata.json`.
