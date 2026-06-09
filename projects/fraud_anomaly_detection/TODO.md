# To-do — fraud_anomaly_detection

Parked items to revisit. Not status, not learnings (see `LEARNINGS.md` for
those) — things we've decided are worth doing and don't want to lose.

## ▶ v3 IS LIVE — START HERE (2026-06-09)

The rebuild is DONE. New dataset **`v2_2ac98b52`** (table
`fraud_advance_feature_base_automl_v3`, 2,412,045 rows × 115 cols, span
2025-01-01 → 2026-06-08; `dry_run=False`). Adds the 3 hashed graph-node keys
(`email_key`/`phone_key`/`address_key`, SHA-256, ~1.38M distinct each; email
still near-noise, max 6 users) + deeper history; drops `name_match_official` /
`official_name`. Full facts + the closed-out graph verdict are in
`docs/HANDOFF.md` — read that first.

**Training/eval pooling (settled):** keep full Jan-2025→now in the base table
(as-of history for the graph/edges), but pool `training_data.sql` rows to
`feature_as_of_ts >= 2025-08-01` AND mature (`label_mature_d45 = 1`) — ~7-month
graph warm-up (no left-censoring) + full 45-day labels (no undefined-label
dilution). `training_data.sql` feeds BOTH train and test; SPLIT_PCT divides them.

**Focused next steps (wendao picks):**
1. Re-baseline on v3 with the Aug-2025/mature pool — confirm locked scenarios
   fire (`validation --no-dry-run`), re-measure precision/volume on deeper
   history, establish the new residual.
2. **★ Graph with the NEW node types** — re-point `analysis/graph_discovery_sweep.py`
   at `v2_2ac98b52`, add `phone_key`/`address_key` node types (types could reach
   5). Multi-type density is what worked; this is the direct test of whether v3
   lifts the v1 ceilings (review-tier, ~0.01% coverage).
3. **Bad-neighbour-count as a model feature** — better-populated now (Aug start
   matures more seeds). Review-tier as a rule; test as a feature.
4. If graph still caps out → pivot to Tier-3 data for the neobank fast-churn
   cohort the graph can't touch (income/payroll, IP-intel, ACH returns; below).

**What we know works (carry forward):** block-tier (≥90%) = the locked
sharing-edge scenarios only; residual block-tier is exhausted (7 ways); the
durable residual signal is review-tier (~5–7×); graph multi-hop is review-tier +
low-coverage, best as a feature/queue (full verdict in LEARNINGS 2026-06-09).

## ⭐ CONSOLIDATED FEATURE-ADD PLAN (2026-06-08)

The single authoritative list of what we add to the feature base, synthesized
from our own analysis + the team's incident/clawback dashboard, the 3 Tabapay
clawback queries, the combo-detail pull, and `scoring_model_20260429.sql`.
Detailed rationale for each item lives in the sections below and in LEARNINGS.md.

**The aim:** enrich the cleanly-definable features AND replicate the team's own
detection logic (the bank-account-sharing flags behind their ~$2M figure) **as-of**,
so our definitions line up with theirs. Most Tier-1 items already mirror the
team's trigger list / scoring model.

**Build status (2026-06-08):** Tier-1 SQL **written** — old two-step SQL archived
under `data/queries/archive/`; `data/queries/base_table.sql` now inlines the full
upstream + all Tier-1 features (heuristic byte-identical, confirmed; adversarial
review clean on as-of/grain/dedup/column-flow). `config.py` unchanged (new columns
auto-become features; no raw PII emitted). **NOT materialized** — gated on:
due diligence DONE (2026-06-08, live warehouse): all borrowed columns exist;
EXPLAIN compiles end-to-end (2 CartesianJoins = the benign 1-row `params`
CROSS JOINs). Fan-out scanned for every edge — email max 4 users/value (unique),
phone max 427, address max 81 (junk `MAIL RETURN`/`PLEASE UPDATE`/zip 00000-3
screened), persistent-id max 576. **Device blowup found + fixed:**
`user_client_metadata` is SCD (176M rows, ~11/user-device) → `device_links`
deduped to one row per (user,device) on earliest `valid_from`. SQL is correct and was **MATERIALIZED** →
dataset `v1_76d3ad45`, table `fraud_advance_feature_base_automl_v2` (1,021,950
rows × 113 cols; old `_automl` table left intact). Post-build validation:
splits healthy (positives in train AND test, ~80/20); all sharing edges
populated (device/address/phone/persistent/email rows with ≥2 users present);
name_match_first/last avg ~97 and last-name mismatch shows ~1.25× never-paid lift
(real, independent signal). **Two findings from the live data check:**
(a) `name_match_official` was NOISE — `official_name` is the account PRODUCT type
("Checking"/"Varo Checking"), not the holder name → DROPPED from the SQL;
real holder-name match needs the Plaid identity/owner source (Tier-3 pull).
(b) `is_joint` is valid but very rare (~0.03%) — kept as a cheap disqualifier.
NOTE: the materialized `v1_76d3ad45` still contains the `name_match_official`
noise column (SQL fixed for next build); drop it on the next rebuild (batch with
other refinements rather than re-running 3.3TB for one column).
Harness bug found: `data/sources/snowflake.py::_scrub_sql` doesn't strip `/* */`
block comments and an apostrophe in a comment desyncs its literal tracking →
comment `;`/`'` leak into the single-statement guard (worked around by sanitizing
comments; real fix is a library to-do).

Process note: the as-of/grain review + EXPLAIN validate STRUCTURE; only the live
value scan caught the `official_name` SEMANTIC issue — scan column *contents*, not
just existence, before trusting a derived feature.

**Settled ground rules (apply to everything):**
- **As-of, always.** Every count/velocity is anchored to the advance's
  `feature_as_of_ts` (prior-only). We do NOT copy the team's hindsight/global
  counts — theirs are forensic (monitoring/clawback), ours are predictive.
- **`heuristic_fraud_score`/`band` stays byte-identical** (proxy-label
  comparability). The team's newer `scoring_model_20260429` band differs from
  ours; if we want their numbers, add as a SEPARATE column — never replace.
- **No credit-risk model scores** (`neobankxgboostmodelv1score`,
  `LOGISTICMODELV1SCORE`, m1–m4 underwriting joins) — confounds fraud with credit
  risk, which is the project's north star to separate. Excluded. (Confirmed 06-08.)
- **Sentinel screen** for phone/email/address before any sharing group-by.
- **Emit derived counts/scores only, not raw PII** (no raw email/phone/address
  columns) → no `config.py` change; new columns auto-become features.
- Each new scenario records its shape-stat, no-innocent-version argument, window
  rationale, AND a proactive(gate)/reactive(queue) timing tag.

### TIER 1 — this rebuild, NO new data pull (feature SQL over already-joined sources)

| # | Add | Source status | As-of? | Notes |
|---|---|---|---|---|
| 1 | `users_on_persistent_account_id_{72h,7d,30d}` | field already carried | as-of | Mirrors the team's "3+ matched Plaid identities" trigger. |
| 2 | `users_on_device_id_{72h,7d,30d}` | source joined | as-of | ~96% unique → cheap. |
| 3 | `users_on_address_{72h,7d,30d}` | joined-but-dropped | as-of | Sentinel screen; family = innocent → needs disqualifier. |
| 4 | `users_on_phone_{72h,7d,30d}` | joined-but-dropped | as-of | Sentinel screen (dummy phones). |
| 5 | `users_on_email_{72h,7d,30d}` | joined-but-dropped | as-of | Sentinel screen (placeholder emails). |
| 6 | `name_match_first/last` (Jaro-Winkler, entered vs `matched_*`) | joined-but-dropped | as-of | Row-level, cheap. Confirmed by team Chart J. Validate on residual ruler. |
| 7 | `official_name` + holder-name-vs-identity-name match | new column, same source | as-of | Synthetic/stolen-acct tell; also virtual-number collapse key. |
| 8 | `is_joint` | new column, same source | as-of | Disqualifier input, not a standalone feature. |
| 9 | `IS_NEOBANK_HIGH_RISK_INSTITUTION` | ready on `fct_loans` | as-of | **Cheapest high-value add**; ties to Chime combos. |
| 10 | Institution concentration (`institution_id`/`name` as usable categorical) | already emitted | as-of | Currently underused. |
| 11 | Prior-only advance velocity: `min_hours_between_advances`, prior `total_advances_on_account`, `avg_advances_per_month`, `max_users_in_72h` | re-derive as-of | as-of | From `scoring_model_20260429` (theirs is global → re-derive). |
| 12 | Create→advance / funnel speed promoted to first-class columns | timestamps present | as-of | Round-1's strongest blind-spot factor; today only a scenario stopgap. |
| 13 | Socure `decision`/`status` as real categoricals | already pulled | as-of | Only `has_kyc` derived today. |
| 14 | **Team detection flags (as-of):** `flag_3_users_72h`, `flag_5_users_ever`, `flag_10plus_users_ever` | 3-72h & 5-ever flags already in upstream; add 10+ | as-of | **The team's core ~$2M logic** (5+ ever, 3+ in 72h, 10+ for clawback) — replicated as-of to align with their definition. |
| 15 | `ring_label`/`ring_rank` (account ring grouping by $) | derive | n/a | Reporting/case-review, NOT a model feature. |

### TIER 2 — graph / entity-ring detection — EXPLORED on v1, review-tier verdict (2026-06-09)

**Verdict (full detail: LEARNINGS 2026-06-09):** multi-hop entity rings DO find
net-new fraud, but **review-tier (~40–65%) at ~0.01% coverage**, not block-tier.
What works = degree-capped advance-co-occurrence edges + a small dense
**multi-resource-type** component; best durable rule `comp≥5 & types≥2` (~55–65%
stable out-of-time); best use = a model feature / review queue, not a real-time
block. No durable ≥90% multi-ring pocket in the residual. **Open question for v3:
do the new email/phone/address NODES + deeper history lift this?** (▶ START HERE
step 2.) Tooling: `analysis/graph_discovery_sweep.py` + `graph_validate_winner.py`
+ `graph_seed_coverage.py`. Original framing kept below.

The Tier-1 edges are each a **1-hop** view (how many users share THIS row's bank
account / device / persistent-id / phone / ...). The next axis is the **graph**:
link all entities and reason about multi-hop rings, not one shared column at a
time. This is genuinely unexplored here and is the most promising remaining
direction now that single-edge discovery is exhausted (see LEARNINGS 2026-06-08:
no block-tier conjunction left in the residual; the lever is new structure/data).

**The idea (wendao's framing).** Build a graph whose nodes are entities —
`user`, `bank_account_key`, `persistent_account_id`, `device_id`, `address_key`,
`phone`, `email`, maybe `name` — and whose edges connect a user to every resource
it touches. Then derive per-advance features from the user's position in the graph:
- **connected-component size** (how big is the ring this user sits in),
- **number of prior BAD users in the component** (never-paid / DPD45 / clawback
  on PRIOR advances of other users in the same component),
- **distance to a known-fraud node** (hops to the nearest seed-fraud user),
- component density / # distinct resource types shared / growth rate, etc.

**How to approach (directional — not yet a plan):**
- *Construction:* a heterogeneous (multi-partite) graph, user-nodes linked to
  resource-nodes. Connected components over the resource-sharing edges are the
  rings; `network_*` today is a degenerate bank-only proxy of exactly this.
- *Candidate tooling:* start simple and interpretable — `scipy.sparse.csgraph`
  connected components / `networkx` for component stats + BFS distance-to-seed;
  label-propagation for "fraud proximity"; only reach for a GNN if the simple
  graph features underperform. Build the discovery in `analysis/` first
  (read-only on the pinned snapshot), promote the winners into `base_table.sql`
  as columns once they validate — same path the Tier-1 edges took.
- *Seeds for distance-to-fraud:* prior-advance never-paid / DPD45 (our ruler),
  or the heuristic E_L band, or eventually the clawback list (Tier-3 label).
- **THE HARD PART — as-of correctness (non-negotiable, same discipline as the
  edges).** The graph must be built **prior-only per advance**: only nodes/edges
  whose timestamps precede `feature_as_of_ts`, and "prior bad users" must be bad
  *as of then* (an advance that went bad LATER cannot inform this one). A naive
  whole-snapshot graph is massively leaky (this is the exact trap that inflated
  the early device screen — LEARNINGS 2026-06-06/07). Expect this to dominate the
  engineering cost; design the as-of snapshotting before building features.
- *Scope:* likely needs a base-table rebuild (overnight) once the feature set is
  chosen; prototype the graph + features on the pinned `v1_76d3ad45` first to
  prove lift over the 1-hop edges before paying for the rebuild.

(Supersedes the old one-line Tier-2 note; the parked "Graph features" bullet in
the Feature-engineering section below is the same idea, kept for its IP/bank/
address framing.)

### ✅ NEXT SQL REBUILD — DONE 2026-06-09 (= dataset `v2_2ac98b52`)

**Executed.** All items below shipped in `fraud_advance_feature_base_automl_v3`:
history extended to 2025-01-01; `email_key`/`phone_key`/`address_key` emitted as
SHA-256 hashes (sentinel-screened); `name_match_official` + `official_name`
dropped. See the **▶ v3 IS LIVE** section at the top for the dataset facts and
next steps. Original change list retained below for provenance.

The batch of changes run **together** to rebuild the base table for the graph
effort. All confirmed against `data/queries/base_table.sql`. (The graph was
prototyped on the old `v1_76d3ad45` FIRST — verdict: review-tier + low-coverage,
see LEARNINGS 2026-06-09; v3 is the lever to test whether more node types lift
it.)

1. **Extend the history depth (the left-censoring fix — the headline reason).**
   The two params at `base_table.sql:18-22`:
   - `output_start_ts`: `'2025-12-01'` → **`'2025-01-01'`** (emit advances from Jan 2025).
   - `history_start_ts`: `'2025-11-01'` → **push back well before output_start** (decision:
     `'2024-01-01'` or the earliest available) so the cumulative graph + as-of
     windows have real prior depth before the first emitted advance. Today it is
     only ONE month of lookback.
   - *Why:* the Dec-1 floor left-censors the graph — a ring that formed earlier
     looks empty (and a user who went bad pre-floor never appears as a fraud
     seed). Windowed edges only need ~30-90d of warm-up; a cumulative/lifetime
     graph wants as much history as we can afford. Cost: 3.3TB scan scales with
     rows — pick `history_start_ts` deliberately.

2. **Emit stable graph-node KEYS for the joined-but-dropped entities (the
   "we already have it" finding — CONFIRMED).** `identities_one_per_user`
   (`base_table.sql:88-108`) and `bank_account_links` (`113-160`) already JOIN
   `email`, `phone_number`, `matched_street_address/city/zip/state`, and the name
   fields — the final SELECT drops them (only the `users_on_*` counts survive).
   To use them as graph NODES, emit a **stable surrogate/hash key per entity**
   (e.g. SHA256 of normalized email; normalized phone; normalized address tuple)
   — **NOT raw PII**, so no `config.py` metadata/PII change. Apply the **sentinel
   screen** (dummy phones/emails/placeholder addresses) BEFORE hashing. Link
   timestamp for these edges = `identity_created_time` (already present — same
   anchor the existing edges use). Cost class = the device build (derive over
   already-joined sources, no new pull).
   - **Do NOT add `ip_address` as a node:** raw IP sharing is dead (NAT) AND in a
     graph it's actively harmful — one NAT/carrier IP merges thousands of
     unrelated users into one giant junk component. IP stays off the node list.

3. **Drop the `name_match_official` noise column.** Already fixed in the SQL +
   added to `config.exclude_cols`; bakes out on the next materialize. (The
   materialized `v1_76d3ad45` still carries it — see LEARNINGS/HANDOFF.)

4. **Verify `official_name` / `is_joint` source columns** (the `VERIFY` note at
   `base_table.sql:135-136`) — confirm `base_prod__plaid_accounts` carries them.

5. **Library: fix `_scrub_sql` block-comment / apostrophe bug**
   (`automl/data/sources/snowflake.py`) so we can drop the comment-sanitizing
   workaround in `base_table.sql`. (Library to-do, listed below; batch awareness here.)

### TIER 3 — needs a NEW pull (parked, NOT this rebuild)
- **Debit-card / linked-card sharing** + **card/bank connect-attempts** (probing).
- **Plaid transaction history** — volume/$ 1/7/30d, microdeposit exclusion,
  categorized Zelle/Venmo/Cash-app/gambling, payroll/balance/shell-account.
- **KYC-vs-Plaid-holder-name** (Plaid identity product) — `official_name` partly covers.
- **ACH return codes** — reactive: sharper fraud-vs-credit **label** + recyclable
  onto the entity as a forward feature.
- **Tabapay/clawback outcome** (RETURNED / list membership) — strong **label**
  material, closer to ground truth than the DPD45 proxy.
- **Partial-payment recovery + extended DPD (150/365/545)** — for the true NET
  loss figure (the $1M→$2M machinery), not gross.

### Parked / out of scope for this rebuild
- **#6988 Chase virtual account numbers** — we don't have the virtual-card data,
  and it's unsolved on both sides. Not adding now. (If ever revisited: needs a
  fragmentation-resistant key — `routing + official_name` / institution-burst.)

**Decisions settled (2026-06-08):** all counts/flags are **as-of only** (no
current-state forensic columns); credit-risk model scores excluded; #6988 probe
dropped.

---

## Library / harness follow-ups (surfaced during the 2026-06-08 build)

Not fraud-specific — candidate automl-library to-dos noted here so they aren't lost.

- **Dataset id scheme reads as static `v1_*`.** Dataset ids are
  `v{recipe_schema_version}_{content_hash8}` (e.g. `v1_42baf0ba`, `v1_76d3ad45`).
  The `v1` is the *format/recipe* schema version (hardcoded `1`), NOT a per-dataset
  sequence — so successive, genuinely-different datasets all read `v1_*` and the
  "version" never increments, which is confusing (you'd expect v1 → v2 as the data
  changes). Investigate the dataset-id entry point (`automl/data`): consider a
  human-readable sequential/lineage label alongside the content hash, or clearer
  naming, so a new dataset visibly reads as a new version.
- **`_scrub_sql` mishandles comments** (`automl/data/sources/snowflake.py`): it
  strips `--` lines and string literals but NOT `/* */` block comments, and an
  apostrophe inside any comment opens a phantom literal that desyncs tracking, so
  comment `;`/`'` leak into the single-statement guard. Worked around by sanitizing
  comments in base_table.sql; real fix = strip block comments + handle comment
  scanning before literal tracking.

## Feature engineering (parked 2026-06-05)

The round-2 finding (the heuristic's ring signal is ~one feature family)
means model quality is currently feature-limited. The general nugget behind
the circular pattern: **sharing of a scarce resource across many fresh
identities** — bank account today; the same pattern mined from other
columns below. We own the upstream SQL and can recreate/extend the base
table as needed.

- **Device / persistent-id sharing — evidence-backed, ready to build (2026-06-08).**
  The earlier within-snapshot screen ("≥3 users → 81.6% on 69 rows") was
  **leakage-inflated**; honest as-of, prior-only numbers (residual-mature, base
  5.1%): `device_id` ≥2 distinct users within 7d = 34 @ 79%;
  `persistent_account_id` ≥2 within 7d = 7 @ 100%; net-new ≈ 35 rows @ 80%
  (+9% over the rings' ~375). **Shape/abnormality (the justification):** both
  IDs are ~96% unique, so a value shared by ≥2 users is a sub-0.1% anomaly
  (device 0.07%, persistent-id 0.05%, bank_account 0.09% of values shared).
  **Build:** add `users_on_device_id_{72h,7d,30d}` +
  `users_on_persistent_account_id_*` as-of windows to the base-table SQL
  (mirror `users_on_bank_account_*`, anchored on `identity_created_time`), then
  register two scenarios — `ring_device_burst` (`users_on_device_id_7d >= 2`)
  and `ring_shared_persistent_account` (`users_on_persistent_account_id_7d >= 2`).
  Window choice: device decays with lookback (72h=87%, lifetime marginal=28%) →
  7d; persistent-id holds 100% at every window → 7d for compute. Needs a
  base-table rebuild (Snowflake/VPN). Raw IP sharing stays worthless (NAT).
  **Convention going forward:** each scenario's `theory` records its shape stat,
  no-innocent-version argument, and window rationale; retrofit
  `ring_account_reuse` and `ring_identity_burst` likewise (bank 72h≥3 = 89%/16×;
  lifetime dilutes to 57%).
- **More scarce-resource edges — same rebuild, no new pull (added 2026-06-08).**
  The base-table SQL already *joins* the identity table in full but the final
  SELECT drops these columns. So address / phone / email sharing land in the
  **same cost class as the device build** (a `users_on_<key>_{72h,7d,30d}` as-of
  aggregate over already-joined sources + one rebuild) — batch them with device +
  persistent-id rather than building device alone. Shape stat is the same nugget
  (a value shared by ≥2 fresh identities is a sub-0.1% anomaly), except address
  has an innocent version (families) → needs the freshness+count conjunction +
  disqualifiers. **Sentinel screen required** for phone/email/address (see
  implementation notes below). The team's trigger list confirms these edges
  (address ≥3, phone ≥3, same-bank ≥3, 3+ matched Plaid identities).
- **Name-matchiness — evidence-backed, buildable now, no new source (2026-06-08).**
  Jaro-Winkler (better than Levenshtein for names) between entered
  `first_name/last_name` and KYC `matched_first_name/matched_last_name` — both
  columns already joined, just dropped from the final SELECT; it's a *row-level*
  score, not a sharing join, so it's cheap. The team's Slack thread resolved
  toward a real relationship (NO_FRAUD ~100 match vs EXTREMELY_LIKELY 46–66).
  **Caveat:** validate against never-paid-DPD45 on the residual/LOW band (our
  ruler), not against the heuristic band — the band is built from ring features,
  so correlating with it partly just re-confirms ring co-occurrence. Down-weight
  common names (signal quality, not compute). This is the *internal* name match;
  KYC-vs-Plaid-holder name is a separate, new-source item below.
- **Speed of monetization** — time from signup → bank link → first advance;
  whether the first action maxes the available amount. Rings move fast;
  real users meander. (Timestamps largely present in metadata.)
- **Bank-account quality** — account age at link time, name match between
  bank holder and identity, deposit history depth (payroll present vs empty
  shell account).
- **ACH return reason codes — high-value, call-out (emphasized 2026-06-08).**
  R10/R05 unauthorized = fraud-shaped vs R01 insufficient funds = credit-shaped.
  Two distinct uses, and the *timing* matters: the return arrives **after** this
  advance, so it can never prevent the advance it's on (reactive, not proactive).
  But (a) it is prime **label** material — sharper than DPD45 at separating
  first-party fraud from credit stress — and (b) it **recycles into a proactive
  feature**: an R10 on this bank account / device / persistent-id becomes a
  forward-looking signal for that entity's *next* advance (same trick the
  prior-advance-velocity features already use). Carry a timing tag so reviewers
  know a rule built on it is a queue-builder, not a gate. Needs the ACH-returns
  table (new pull).
- **Graph features** — connected-component size/growth over shared
  device + IP + bank + address; generalizes the ring signal beyond one
  bank account (the existing `network_*` columns are aliases of the
  bank-account count, not a real graph).

## Cross-check vs the incident/clawback dashboard — Incidents #6848 & #6988 (2026-06-08)

Compared our upstream feature SQL against the team's 3 Tabapay clawback queries
and the Mode dashboard. Key conclusions and net-new items:

- **Their count is forensic, not predictive — do NOT copy it.** The clawback
  queries source `base_prod__plaid_accounts_current_state` and use an unbounded
  `COUNT(DISTINCT user_id) OVER (PARTITION BY routing, account)` (lifetime,
  current-state, counts future-attached users). Correct for *recovering* money
  with hindsight; **leaky if used as a feature**. Our as-of windowed
  `users_on_bank_account_*` is the right construction for prediction — keep it.
  Their ~$2M is a hindsight figure, not a prediction target.
- **`persistent_account_id` sharing is the likely antidote to #6988 (Chase
  virtual account numbers) — promote it to headline, VERIFY FIRST.** #6988 evades
  the team's detector because each virtual number is a distinct `account_number`,
  so `COUNT(DISTINCT user_id) PARTITION BY routing, account` never accumulates
  past 1. The fix is to count sharing on a key the fraudster can't fragment.
  Plaid's `persistent_account_id` is *meant* to map virtual numbers back to the
  real account — which would catch the ring `account_number` counting misses.
  **Empirical check before banking on it:** confirm `persistent_account_id`
  actually stays constant across Chase virtual account numbers on real data; if
  Plaid also fragments, fall back to `institution_id + official_name` collapse.
- **`official_name` (Plaid account-holder name) — net-new, not in our SQL.**
  Pull it from the plaid-accounts table (the clawback query reads `pa.official_name`).
  Two uses: (a) holder-name-vs-identity-name mismatch (synthetic/stolen account —
  an as-of-safe identity-coherence signal), (b) collapse virtual `account_number`s
  that share one holder name at one institution (the #6988 fingerprint).
- **`is_joint` — net-new false-positive suppressor.** A legitimately joint account
  has multiple holders; use it as a disqualifier on the sharing scenarios so a
  2–3-user joint account doesn't read as a ring.
- **Institution-level burst** — a burst of fresh `account_number`s at one
  institution sharing a holder name/identity is the virtual-number ring's
  fingerprint; per-account counting can't see it. Candidate scenario once
  `official_name`/`institution_id` features exist.
- **Jaro-Winkler name-match confirmed independently** by the dashboard's Chart J
  (spread of Jaro-Winkler by fraud band) and the Anders Zhou thread — reinforces
  the buildable name-match item above. As-of-safe (both names known at
  origination), so replacement-grade, not merely additive.
- **The operational layer they use** (`base_prod__payments`,
  `base_prod__payment_methods`, `base_prod__tabapay_card_transactions`) is the
  source for the ACH-return / card-clawback signals already parked below.

### From `scoring_model_20260429.sql` (the parent of our upstream) — reviewed 2026-06-08

- **Their whole scoring apparatus is hindsight/global, not as-of** — the
  `users_on_bank_account` / `max_users_in_72hr` / advance-velocity counts have
  only a `created_at >= '2025-12-01'` floor, no advance-time bound. Correct for
  monitoring/clawback, leaky for prediction. Our as-of version is the predictive
  contribution — do not adopt their counting verbatim.
- **Band formula diverged from ours.** Their newer model dropped `network_score`
  (component 5 → `avg_advances_per_month`) and changed the velocity term
  (component 4 → `advance_days_span <= 1/3` with ≥2 advances, vs our
  `prior_advances_72h/7d`). **Keep `heuristic_fraud_score` unchanged** (proxy-label
  comparability). If we want to track the team's current numbers, add their
  formula as a SEPARATE column, don't replace ours.
- **`IS_NEOBANK_HIGH_RISK_INSTITUTION` — adopt (high value, low cost).** A
  ready-made flag on `fct_loans`, as-of-safe (bank property at origination),
  directly ties to the Chime combos. Add it.
- **Account advance-velocity as prior-only re-derivations:**
  `min_hours_between_advances` (min pairwise gap, via LAG window),
  `total_advances_on_bank_account`, `avg_advances_per_month`,
  `max_users_in_72hr` — all currently global; re-derive as-of.
- **`ring_label` / `ring_rank`** — account-level ring grouping by total $ (base-26
  label). Useful for case-review/reporting, not a model feature.
- **Partial-payment recovery + extended DPD** — `Gross_w_partial_{7..1000}` and
  DPD/mature at 150/365/545. The true NET loss (vs gross); likely part of why the
  figure moved $1M→$2M. Connects to the recovery finding already in LEARNINGS.
- **Use with caution — credit-risk scores** (`neobankxgboostmodelv1score`,
  `LOGISTICMODELV1SCORE`, the m1–m4 underwriting joins). Pulling these into fraud
  features risks confounding fraud with credit risk (the project's north star is
  separating them). Leave out unless deliberately wanted.
- **`ASOF JOIN`** (Snowflake native) for KYC/device — cleaner than our
  QUALIFY-based as-of; technique upgrade for the rebuild.

## Not yet available — needs a new pipeline/source (from team trigger list, 2026-06-08)

We don't have these in the base table or its joined sources today; each needs a
new pull before it can be a feature. Parked here so the team's prior work isn't
lost. All are **proactive** (knowable at advance time) once pulled, except where
noted.

- **Debit-card / linked-card sharing** — "same debit card on 3+ accounts",
  "linked cards in 1/7/30d". A whole scarce-resource edge missing from the base
  table; needs a card-link source. Pairs naturally with the device/address/phone
  batch once the card data exists.
- **Card / bank connect-*attempts* (1/7/30d)** — probing behavior, distinct from
  successful links (`bank_accounts_per_user_asof` counts successes only). Needs
  connection-attempt logs.
- **Plaid transaction history** — txn volume / $ in 1/7/30d, microdeposit
  exclusion, observation-week handling; account balance / deposit depth / payroll
  presence → shell-account detection; **categorized P2P/gambling** (high-volume
  Zelle / Venmo / Cash-app / gambling = cash-out/mule-shaped). The single biggest
  net-new block on the team's list. Needs the Plaid transactions pull.
- **KYC vs Plaid bank-holder name mismatch** — the team's "KYC does not match
  Plaid Identity". Distinct from the *internal* name-match above (which is free);
  this needs the Plaid identity product (the bank account holder's name).
- **IP intelligence — DERIVED IP signals (not raw IP sharing).** Confirmed
  2026-06-08 (`analysis/ip_screen.py`): raw IP *sharing* is dead even at the
  leaky current-state ceiling (ip_address users>=2 -> 0.9x; >=10 -> 0.9x; NAT and
  shared household IPs swamp it) — do NOT build a users_on_ip edge. The
  fraud-shaped IP signals need an enrichment source: (a) **datacenter / hosting /
  VPN / proxy flag** (a real borrower does not advance from AWS/DigitalOcean/a VPN
  exit) via an ASN/hosting-range or IP-intel DB (MaxMind GeoIP2-ISP/ASN, IPinfo,
  IPQualityScore); (b) **geo mismatch** — IP-geolocated state/metro vs the KYC
  address state/zip or the phone area code (GeoLite2). Both are proactive once
  pulled. Faint freebies already in the table: signup_ip==latest_ip (1.7x),
  has_ip_address==0 (2.3x, tiny/confounded) — modifiers, not scenarios.

## Methodology note (flag, don't silently resolve) — 2026-06-08

The team's "Scorecard (re)design / scorecard sharing" line is the **additive
points** approach — that is literally today's `heuristic_fraud_score`/`band`.
This project deliberately moved to **conjunctive scenarios** (SCENARIOS.md:
additive weights are arbitrary and ignore dependencies — high amount is
protective alone, 8–14× dangerous inside newness). Before building toward the
team's scorecard framing, confirm we're aligned on stance rather than quietly
maintaining both.

## Implementation notes — shared-resource base-table rebuild (parked 2026-06-08)

Refinements to handle during/after the rebuild, captured so they're not
rediscovered later (the build will ship a sensible default for each):

- **Sentinel screen before any sharing group-by (phone/email/address).** The
  cost/correctness risk is fan-out from dummy values (e.g. `0000000000`,
  `noreply@`/`test@example.com`, placeholder addresses) attaching to thousands of
  users. Null them out *before* the count: empty/placeholder emails, sub-10-digit
  or repeated-digit phones, addresses missing street-or-zip. Device/persistent-id
  are ~96% unique → no screen needed. Default lists ship in the SQL; refine the
  dummy lists against real value-frequency once built.
- **Giant-group cap (refinement).** Even after the sentinel screen, a real value
  shared by an absurd number of users is more likely a data artifact than a ring
  — consider excluding/capping groups above some size. Not in the first build.
- **Email canonicalization (refinement).** gmail dot/plus-alias folding for the
  *sharing* key — deferred; first build keys on lowercased/trimmed email.
- **Name-match is signal-quality, not compute** — row-level Jaro-Winkler, no
  fan-out. Down-weight common names; validate on the residual ruler (see above).
- **Inlining requires raw-table read grants.** The new base SQL reads
  `fct_loans`, `base_prod__*`, `user_client_metadata`, etc. directly (no longer
  just the pre-built `fraud_advance_feature_base`). Confirm the harness Snowflake
  role has those grants at preflight, or the materialize step fails there.
- **Register gains a proactive/reactive timing tag** — alongside the existing
  `theory` / shape-stat / window convention, record whether a scenario is a gate
  (proactive, decision-time) or a queue-builder (reactive, e.g. ACH-based).

## Other parked

- **Withhold experiment** — refit anomaly models excluding the
  heuristic-component family (`users_on_bank_account_*` + aliases) to see
  whether the rest of the feature space independently finds the same
  frauds. Shelved 2026-06-05: not ready to set those features aside while
  they're genuinely indicative.
