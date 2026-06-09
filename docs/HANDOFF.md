# Handoff — continue here

Temporary hand-over note: where the last session left off and what's open, so a
new session can pick up. **Not a changelog** — keep only what's current and
relevant; detail lives in the project docs. Rewritten at each wrap, not appended
to.

**Last updated:** 2026-06-08 (unsupervised-on-v2 + scenario lock + graph direction
+ finalized next-SQL-rebuild scope)

## How to pick up (per wendao)

Don't dive into code. (1) Read this plus the project docs below; (2) summarize
where things stand; (3) **recommend 2–3 options and let wendao pick.** The next
move is wendao's call, not a queue to drain.

Project docs (`projects/fraud_anomaly_detection/`): **`LEARNINGS.md`** (start with
the newest 2026-06-08 entry — "unsupervised on the v2 features"); **`TODO.md`**
(the ⭐ CONSOLIDATED FEATURE-ADD PLAN, and the expanded **TIER 2 — graph/entity-ring
detection** which is the next major effort); `SCENARIOS.md`; `scenarios/register.yaml`.

## What this session did (2026-06-08)

Goal was wendao's: build a model on the new v2 feature base, see if it finds fraud,
what patterns emerge, bucket into scenarios. wendao chose **unsupervised** (anomaly
shape ≈ fraud intent; supervised on DPD45 blends in credit risk).

- **Feature due diligence first** (`analysis/feature_due_diligence.py`) on the full
  v2 dataset **`v1_76d3ad45`** (1.02M rows, in the NON-dry-run scope). All 23 new
  Tier-1 columns are as-of safe; all outcome/label columns are non-feature. **Only
  change needed: excluded `name_match_official`** (product-type noise) — added to
  `config.py` exclude_cols AND dropped in the analysis feature space (the stored
  registry is baked at materialize time, so config only bakes out on next rebuild).
- **Unsupervised (Isolation Forest + GMM) on the gated residual** (`unsupervised_lens.py`):
  as a GLOBAL ranker it's still flat (~1.3× AP, exactly round-3) — but it
  **independently rediscovers the new sharing edges 100–200× enriched at the top**,
  confirming them as real abnormal-shape signals (deployment vehicle is a rule, not
  the score). Per-edge precision in `edge_precision_screen.py`.
- **Locked 2 new draft block scenarios** (register `2026-06-08.2`, evidence refreshed
  on the full snapshot via `validation --no-dry-run`, tests pinned):
  `ring_shared_persistent_account` (persistent 72h≥2, 92% never-paid, the #6988
  virtual-number antidote) and `ring_device_burst` (device 72h≥3, 88%, the real
  net-new contributor at unique_n=232). **Evaluated `ring_shared_phone` (97%) but
  DROPPED it** — unique_n=2, fully redundant with the other rings.
- **Next-layer / narrow / subgroup discovery** (`residual_next_layer.py`,
  `subgroup_discovery.py` — the "proven algorithm", beam search + held-out
  validation + significance). **Confirmed three independent ways: NO block-tier
  conjunction remains in the residual.** Best remaining is a review-tier
  **neobank × early-tenure × small-amount** cohort (~4–6×, fraud-smelling but ~30%
  precision = a review/mitigate queue, not a block). The ceiling is a feature-space
  limit, not a search/tuning limit.
- **IP verdict** (`ip_screen.py`): raw IP *sharing* is dead (≤1× even at the leaky
  ceiling — NAT/households). The fraud-shaped IP signals are DERIVED (datacenter/
  VPN/proxy, geo/area-code mismatch) and need an enrichment pull (Tier-3).
- **Institution** (`institution_screen.py`): the signal IS Chime (= the neobank
  flag); no sharper institution, Chase not a concentration. The shell-vs-real
  discriminator that would lift the neobank cohort is **income/payroll** (Tier-3).
- **Cleanup:** removed 4 superseded analysis scripts (`conjunction_discovery.py`
  and the 3 pre-materialization screens `asof_sharing_screen` / `asof_breakdown` /
  `current_data_screen` — their job is done now that edges are materialized).

## What's open — wendao to pick

- **★ THE NEXT EFFORT — graph / entity-ring detection, ON THE EXISTING DATASET (NO
  new SQL).** This is the priority and the framing matters: wendao wants a **new way
  to use the same data**, not another feature pull (there is always another feature —
  that is a separate, deferred track below). Build a graph over the entity keys
  ALREADY in `v1_76d3ad45` — `user` · `bank_account_key` · `persistent_account_id` ·
  `device_id` (· `ip_address`) — and derive connected-component size, # prior BAD
  users in the component, distance-to-known-fraud, component density/growth, etc.
  Generalizes the 1-hop edges to multi-hop rings; the most promising direction now
  that single-edge discovery is exhausted. **email / phone / address / name are NOT
  in the dataset** (only their counts were emitted; raw keys dropped as PII) — adding
  them as nodes is the only part that would need a SQL emit, so it is OUT of scope
  for v1. The next session should **talk through approaches + pros/cons + how-to +
  data-size feasibility (1.02M advances, ~750k users/accounts/devices) + the as-of
  correctness problem BEFORE coding** — see TODO.md TIER 2. Prototype read-only in
  `analysis/` on the pinned snapshot. (Hard part = as-of: build prior-only per
  advance; a whole-snapshot graph is massively leaky.)
  - **Direction settled with wendao (2026-06-08):** start at the BASIC rung —
    **windowed connected-component features** on the existing `v1_76d3ad45`
    (drop a ~30d warm-up for the Dec-1 left-censoring), prove lift over the 1-hop
    edges, THEN escalate only if it pays: distance-to-known-fraud → label-prop →
    (never) GNN. Nodes = `user` · `bank_account_key` · `plaid/persistent_account_id`
    · `device_id`; edge link-time = `identity_created_time`; advance clock =
    `feature_as_of_ts`. **IP is NOT a node** (NAT → giant junk component).
  - **A finalized NEXT-SQL-REBUILD scope is now parked** (TODO.md "★ NEXT SQL
    REBUILD"): extend history (`output_start_ts`→2025-01-01, `history_start_ts`
    pushed back from its current 1-month lookback), emit hashed graph-node keys
    for the **joined-but-dropped** email/phone/address (confirmed sitting in the
    CTEs, dropped at final SELECT — only counts emitted today), drop
    `name_match_official`, verify `official_name`/`is_joint`. Prototype the graph
    on the pinned snapshot FIRST; this rebuild is what the cumulative-graph /
    distance / production version needs.
- **Promotion gate (small, anytime):** wire the 2 locked scenarios into
  `scenarios/backtest/monthly_backtest.py` and run the month-over-month backtest (the
  sign-off stat) to move them draft → signed_off.
- **DEFERRED data-pull track (NOT the next focus — wendao explicitly wants to avoid
  touching SQL for now).** When/if a rebuild happens near sign-off, candidates:
  drop `name_match_official`; `persistent_account_id` 90d/lifetime windows;
  **income/payroll** (the shell-vs-real discriminator — would split the ~30% neobank
  cohort that we can't separate today: shell account w/ no real income vs genuine new
  user; strong indirect support, unmeasured); **IP-intelligence** (datacenter/VPN/geo).
  All parked in TODO Tier-3 — do NOT pull these into the graph effort.
- **Optional:** register the neobank review-tier cohort as a `mitigate`/`review`
  scenario, or keep it as a model feature only.

## Gotchas

- **Dataset scope split:** the full v2 build `v1_76d3ad45` (new edges) lives in the
  **NON-dry-run** scope — use `use_project(..., dry_run=False)`. The dry-run scope
  still holds the OLD `v1_42baf0ba` (no new edge columns).
- **The register now requires `v1_76d3ad45`:** new scenarios trigger on the new edge
  columns, and the engine **raises on a missing column**. So `validation` must run
  with `--no-dry-run`, and any consumer/gate must use the full dataset.
- **`config.exclude_cols` edits only take effect on the next materialize** (the loaded
  registry is read from GCS, baked at materialize time — not re-derived from config).
- `experiment run` / `data materialize` preflight needs **Snowflake/VPN** (~30s).
- `--instruction` is **shell-evaluated** by the loop's context-render — keep it plain
  prose (no `->`, parens).
- A killed `experiment run` holds the session lock (~6h self-expiry); release via
  `trial lock release` (ids in `.cache/automl/tmp/session_locks/*.lock/`).
- `base_table.sql` comments must avoid `;` and `'` until the `_scrub_sql` harness bug
  is fixed (TODO.md library follow-ups).

## Loose ends

- **`v1_76d3ad45` still carries the `name_match_official` noise column** (excluded in
  config for the next build; dropped in the analysis feature space meanwhile).
- **Nothing committed this session** — register edit, `config.py` exclusion, the 4
  test edits, the analysis suite (7 scripts, all untracked), and the LEARNINGS/TODO
  updates are all in the working tree. `main` is local-only ahead of `origin/main`.
- Analysis suite kept: `feature_due_diligence`, `unsupervised_lens`,
  `edge_precision_screen`, `residual_next_layer`, `subgroup_discovery`, `ip_screen`,
  `institution_screen` (+ prior `ceiling_probe`, `supervised_lens`, `rule_discovery`).
