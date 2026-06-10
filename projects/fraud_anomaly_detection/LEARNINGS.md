# Learnings — fraud_anomaly_detection

High-level takeaways only: what worked, what didn't, what surprised us.
Append as they emerge; date each entry. (Long-term these belong in MLflow at
the experiment/project level — this file is the ad-hoc home until the
workflow settles.)

## 2026-06-09 — persisted entity-graph store: capability proven on the sample (infra, not metrics)

**What was built (design: docs/superpowers/specs/2026-06-09-fraud-entity-graph-store-design.md).**
The graph effort's throwaway in-memory UnionFind is replaced by a persisted,
self-contained DuckDB store (`graph/build.py`: uncapped edges + full
timestamps + all 7 entity types + full advances snapshot; rebuild-only
refresh) with parameterized igraph views (`graph/load.py`: layers / degree
cap / as-of / dynamic scenario overlay) and question-level helpers
(`graph/queries.py`: near_flagged, components, ring, project_users,
hub_report). Lossless store, opinionated views: every judgment call is an
analysis-time parameter — high-degree entities are STORED in full (they're
the fraud-farm signal; the cap is only a traversal view choice). Deps live
in the project-scoped `fraud` dependency group (`uv sync --group fraud`).

**Capability demo on the 20k fraud-enriched sample** (graph_store_demo;
sample is graph-thinned — numbers are capability evidence, NOT transferable):
build 123,252 edges / 19,301 users across all 7 entity types in seconds;
dynamic scenario overlay flags 1,024 users against the current register at
load time (nothing persisted); 36 users sit within 3 user-hops of a flagged
user (all at hop 1 — proximity queries work; the thinned sample has no
deeper chains); 238 multi-type components (>=3 users & >=2 types, cap=20)
hold 538 fraud users / 1,397 total (~38% vs ~3.9% sample base — the v1
multi-type-density discriminator visibly survives even sample thinning);
hub report cleanly separates farms from infrastructure on the time axis
(top: a 136-user device over 99 days at 38% fraud-user rate vs bank-account
hubs with 40-41 users in 4-9 DAYS at 100% — the latter is the programmatic
fraud signature); ring deep-dive + weighted user-user projection (4,823
pairs at cap=20; 1,419 multi-type) round out the question list. Store
persists, reopens cold, rebuilds idempotently.

**DuckPGQ probe verdict: SKIPPED — extension unavailable on this platform.**
No osx_arm64 build published for DuckDB v1.5.3 (HTTP 404 from the community
extension repo). The probe script (`analysis/graph_pgq_probe.py`) is ready
to produce an AGREES/DISAGREES verdict vs the igraph baseline; retry when
duckpgq publishes for the platform/version, or on a linux/amd64 host. The
platform lag is itself a maturity datapoint: igraph stays the only
traversal engine, as the design assumed.

**Next (unchanged from the v3 plan):** re-point the build at `v2_2ac98b52`
(full v3) — the store schema and views carry over as-is; the value question
(does multi-type density + new node types move coverage off ~0.01%?) is
TODO #2, now answerable with persistent infrastructure instead of one-off
scripts.

## 2026-06-09 — graph / entity-ring detection on v1_76d3ad45: real multi-hop signal, but block-tier is anecdotal + maturity-censored; durable win is review-tier

**The effort.** First real graph build (TODO TIER 2), all on the pinned
`v1_76d3ad45`, NO new SQL. Nodes = `user` + resources `device_id` /
`bank_account_key` / `persistent_account_id` (NOT `ip_address`: NAT/households
make it a giant-junk-component generator). Edge = advance co-occurrence
(user touched resource at `feature_as_of_ts`). Connected components generalise
the 1-hop `users_on_*` edges to multi-hop rings. Six read-only scripts:
`graph_component_screen` (windowed CC), `graph_edge_study` (edge-event gap),
`graph_giant_component` (junk-node cap), `graph_seed_proximity` (distance-to-
known-bad), `graph_discovery_sweep` (consolidated battery), `graph_validate_winner`
(concentration + out-of-time). All strictly as-of / prior-only, degree-capped.

**As-of construction (the trap, handled).** Strictly-prior cumulative graph via
incremental union-find in time order (cheap, O(N alpha)). KEY BUG caught and
fixed mid-effort: day-bucketing dropped same-day advances, and fraud rings burst
intra-day -> 72h comp>=3 came out as ZERO (impossible if generalising the device
ring). Fix = process each day's advances in timestamp order, linking same-day-
EARLIER ones (still strictly prior, no leak). Self-test asserts this.

**Edge sparsity hypothesis was WRONG (data corrected it).** Advance-co-occurrence
is NOT sparse: devices the SQL flags as >=3 identities/7d carry a MEDIAN of 19
advancers. The real problem is the OPPOSITE -- giant junk components (max 3,789
users) from shared-infra device/bank values (top device = 386 users). A degree
cap of 10-50 (drop ~35-70 promiscuous nodes) gives IDENTICAL net-new results
(genuine rings don't route through junk); cap=5 over-cuts. Persistent-id is
cleanest (max 49 users).

**Discriminator = small DENSE MULTI-TYPE component, not big components.** Big
single-type = junk; a user webbed across device AND account AND persistent-id is
the ring. `comp>=3 & types>=3` -> 64% never-paid (12.6x) vs `comp>=3` (any type)
21%.

**Seed-proximity (distance-to-known-bad) -- the sharpest lever, with two
corrections the data forced:**
- **Seed on bad OUTCOMES (DPD45-matured, activated at `expected_dpd45_date`), not
  scenario flags.** Scenario-proximity is DEAD (~7%, base rate) -- the scenarios
  already took the precise core, so their residual neighbours are the innocent
  excluded tail.
- **Exclude self.** Own prior bad advance is repeat-defaulter / CREDIT history
  (304 rows @ 31.6%, 6.2x -- confounds the north star). The true RING signal is
  proximity to OTHER bad users, and removing self made it SHARPER: dist-1 to
  another bad user = 50% (9.9x), and # other-bad-in-component is the precision
  dial: >=1 -> 49%, >=2 -> 70%, >=3 -> 79%.
- **Sharpest pocket:** `nb_comp>=2 & types>=3` -> **100% never-paid, n=15,
  p=3.8e-20, fully net-new** (no single 1-hop edge fires).

**BUT validation killed the block-tier claim two ways (why this is review-tier,
not a >=90% gate):**
1. **Concentration -- the 100%/n=15 gem is ONE ring** (3 users, 15 advances over
   6 days). All the >=75% pockets are 1-3 distinct rings = anecdotes, not rules.
   Only `nb_comp>=1` (49%, review-tier) spans many (27 distinct rings).
2. **Out-of-time decay -- CORRECTED 2026-06-09 (`graph_seed_coverage.py`).**
   Seed-based rules collapse early->late: nb_comp>=3 100%->56%, nb_d1>=2
   100%->50%, nb_comp>=1 77%->40%. My first explanation ("recent advances are
   seed-STARVED by ~45-60d DPD45 maturity") was WRONG and backwards: late
   advances have a longer prior history, so MORE seeds (504 matured before the
   early median vs 17,555 before the late median, 35x), and fire on MORE rows
   (53 vs 17). The real cause is **small-n early optimism regressing to the true
   rate**: early fires on 17 rows across only 4 rings (76%), late on 53 rows
   across 25 rings (40%). So **the durable, out-of-time-honest precision of
   bad-neighbour proximity is ~40% (~7x), review-tier** -- the headline 50% /
   9.9x was a few-ring overestimate. A genuine but SEPARATE deployment caveat
   remains: the freshest entities' neighbours have not matured, so the feature
   undercounts for brand-new advances (favours older entities) -- a reason it
   suits a review/clawback queue, but NOT the cause of the measured decay.
3. **Coverage is tiny (the volume reality).** On the 537,150 matured eval rows
   (61.8% of residual+warmup; immature rows correctly excluded so precision is
   not deflated), EVERY graph signal fires on ~0.01% of transactions:
   nb_comp>=1 = 0.013% (70 rows), nb_comp>=3 = 0.0035% (19), structural
   comp>=5&types>=2 = 0.0058% (31). Block-tier or review-tier, the graph touches
   ~1 in 7,700 advances. This is the hard limit on the whole direction's value.

**The durable, real-time-usable win = STRUCTURAL multi-type ring (no seed -> no
maturity lag).** `comp>=5 & types>=2` is STABLE out-of-time (61% early / 54%
late) across **7 distinct rings**, ~12x lift, net-new (n=31). `comp>=3 & types>=3`
holds 61%/71%. This is the genuine net-new graph contribution: a ~55-65%
review/block-adjacent rule on a NEW axis (multi-hop multi-resource ring
structure) that the 1-hop edges and scenarios miss and that does NOT decay.

**Verdict (consistent with the whole project arc, now via the graph route).** No
durable >=90% multi-ring pocket exists in the residual -- confirmed a SEVENTH
independent way. The graph adds (a) a stable ~55-65% structural multi-type-ring
review rule (deployable, net-new), and (b) a bad-neighbour-count FEATURE whose
durable out-of-time precision is ~40% / ~7x (as-of-clean; best for a
review/clawback queue or model input, not a real-time block). BOTH cover only
~0.01% of transactions (finding 3) -- real and net-new, but vanishingly low
volume. Block-tier precision still lives only in the sharing-edge scenarios
already locked. Recommendations for wendao (NOT acted on):
register `ring_multitype_structural` (comp>=5 & types>=2) as a review-tier
scenario; add bad-neighbour-count as a model feature; the planned rebuild (deeper
history + email/phone/address NODES) is the lever to grow both -- more node types
directly feeds the multi-type discriminator, and deeper history matures more
seeds. Register/rebuild left to wendao (editing the register mid-comparison
breaks comparability; rebuild is user-gated).

## 2026-06-08 — unsupervised on the v2 features: score still flat, but it rediscovers the new edges; rules win

**Setup.** First analysis on the full v2 build (`v1_76d3ad45`, 1,021,950 rows —
note: the full build lives in the NON-dry-run scope; the dry-run scope still
holds the old 107k `v1_42baf0ba`). Feature due diligence first
(`analysis/feature_due_diligence.py`): all 23 added columns are the Tier-1
features, all as-of safe; every outcome/label column is non-feature; the only
contaminant was `name_match_official` (the known product-type noise) — dropped in
the analysis feature space AND added to `config.exclude_cols` (the stored registry
is baked at materialize time, so the config edit only bakes out on the next
rebuild). Active unsupervised feature space = 63 num+bool features.

**Unsupervised stays closed as a global ranker — confirmed on the NEW features.**
`analysis/unsupervised_lens.py`, gated residual, residual+mature test (136,527
rows, base never-paid 5.04% / DPD45 5.79%): IF full-space AP 0.075 (1.29x),
IF ring-family-withheld 0.076 (1.31x), GMM 0.075 (1.29x) — all sitting exactly
where round-3 landed (IF 0.0751 on the old features). Adding the edges did NOT
revive the anomaly score as a ranking instrument; the aggregate AP is unmoved
because the edges are too rare (sub-0.1%) to shift it and the cohort bulk is the
same fast-cycling-on-fresh-accounts pattern (identity age ~1h vs ~463 days base,
plaid-account age 0d vs 167d).

**But the anomaly model independently REDISCOVERS the new edges at the top.** The
top-0.5% anomaly cohort is enriched 100-200x for every sharing edge (device 171x,
persistent 200x, address 126x, phone 171x, email 155x) and runs ~13-16% never-paid
(2.6-3.3x base), ~66-90% LOW band (heuristic-missed). The ring-family-WITHHELD run
(B) surfaces the same edges (device 121-199x, persistent 200x) — so the edges
carry discovery signal independent of the bank-account ring the heuristic already
owns. GMM's top-0.5% is the sharpest (17.6% DPD45 precision, 3.0x) and also pulls
the neobank flag (2.0x). So: anomaly-for-discovery WORKS for the new edges (unlike
round-3's residual), but the SCORE is not the deployment vehicle — a rule is.

**Per-edge precision (`analysis/edge_precision_screen.py`, residual+mature 685,993
rows) — the registerable shape-stats. All rows are register-invisible (residual),
i.e. net-new beyond the two existing rings:**
- **persistent_account_id — the standout, block-tier, no innocent version.**
  72h>=2 → 93.0% never-paid (18.3x, n=215); 7d>=2 → 89.4% (n=226); 72h>=3 → 98.4%
  (n=189). Holds across windows (same real Plaid account + >=2 fresh identities).
  This is the #6988 (Chase virtual-number) antidote the TODO predicted.
- **device_id — block-tier at >=3** (>=2 dilutes to innocent stale reuse): 72h>=3
  → 80.5% (15.8x, n=343); 7d>=3 → 77.1% (n=376); 72h>=2 only 64.0%, 7d>=2 58.0%.
- **phone — block-tier at >=3 only:** 72h>=3 → 97.0% (19.1x, n=99); 7d>=3 → 82.3%
  (n=124); >=2 is 47-57% (sentinel screen leaves some dummy-number noise).
- **address — review-tier:** 7d>=2 → 21.6% (4.3x); needs >=3 + non-joint
  disqualifier (72h>=3 → 59% but n=22). Families are the innocent version.
- **email — noise-tier:** 7d>=2 → 15.6% (3.1x). Shared/family emails dominate; drop.
- **name_match_last — does NOT carry standalone signal at scale:** <80 → 5.7% vs
  5.09% base = **1.1x** (the earlier "1.25x" was on 107k). At best a weak modifier
  inside a conjunction, NOT a scenario. Honest down-grade.
- **is_neobank_high_risk — broad modifier, not a scenario:** 11.7% never-paid
  (2.3x) but over 102,165 rows (15% of pop). Keep as a model feature, not a rule.
- **Union device|persistent|phone|email (7d>=2): n=595, 56.0% never-paid (11.0x),
  333 never-paid caught** — roughly DOUBLES the rings' ~334 never-paid capture on
  an independent axis. device|persistent only: n=566, 57.8%, 327 caught.

**Takeaway / recommendation.** The path is exactly the project stance: unsupervised
is discovery-only (and here it earned its keep by independently confirming the
edges), the product is precise rules. Candidate draft block scenarios:
`ring_shared_persistent_account` (persistent 72h>=2), `ring_device_burst`
(device 72h>=3), `ring_shared_phone` (phone 72h>=3); address as a review-tier rule
with disqualifiers; drop email and standalone name-match; keep neobank as a feature.
Not auto-registered — pending wendao sign-off + monthly-backtest promotion gate
(editing the register mid-comparison breaks comparability). Tooling all pinned to
`v1_76d3ad45`, dry_run=False, read-only: `feature_due_diligence.py`,
`unsupervised_lens.py` (runs A/B/C), `edge_precision_screen.py`.

**Conjunction discovery (was `analysis/conjunction_discovery.py`, removed in the
2026-06-08 cleanup — superseded by `subgroup_discovery.py` + `residual_next_layer.py`;
findings preserved here) — three findings:**
(1) **email is unrescuable at any threshold** (>=3 collapses to n=1; drop it);
**address buys precision only by trading away all volume** (72h>=4 → 62.5% n=8;
>=5 → 100% n=3) — a real-but-tiny edge. (2) **Create->advance speed does NOT
rescue the weak/>=2 edges** — device 72h>=2 alone 64% vs AND id->adv<=1h 66%;
persistent 93%→94%; email 15.6%→16%. The edge rows are ALREADY fast/fresh
(speed and the edge are collinear in this population), so speed is not an
orthogonal lever ON TOP of a sharing edge — the lever is the count/window
(device wants >=3, not >=2). Speed is the instrument for the broad fast-churn
cohort, not a within-edge separator. (3) **Model-assisted (shallow tree, fit on
train, precision validated on held-out test):** the one clean fraud-shaped
conjunction it discovers is `days_since_plaid_account_created <= 9.5 AND
users_on_device_id_7d > 2.5 AND not-neobank` → **73.2% never-paid (14.5x), n=71
test (78.7% train)** — i.e. device>=3 on a fresh account, consistent with the
single-edge result. Every other high-volume leaf is the **neobank x small-amount
x low-velocity x fresh-account credit-ambiguous cohort** (29.6% never-paid 5.9x
n=1084; 18% 3.6x n=4136) — the same round-3 fast-small-dollar-churn bucket, which
is review-tier (credit stress overlaps fraud here), NOT a block rule. Caveat:
some lower leaves split on `prior_min_hours_between_advances_on_account` at
~247 — that is the NaN-imputed median (12.6% null), an imputation artifact, not
signal. **Net:** the strong sharing edges are self-sufficient at the right
count/window (conjunctions don't strengthen them); the only genuinely additive
"combination" signal is the credit-ambiguous neobank/velocity cohort, which is a
review queue, not a block. A non-greedy rule-miner (skope-rules / RuleFit) could
mine combinations the greedy tree's first-split ordering misses — parked option.

**LOCKED three new draft block scenarios (register version 2026-06-08.1; evidence
refreshed on the full v1_76d3ad45 via `validation --no-dry-run` — the new edges
only exist in the full scope and the engine raises on a missing column, so the
register now requires v1_76d3ad45):** `ring_shared_persistent_account` (persistent
72h>=2, disqualify is_joint), `ring_device_burst` (device 72h>=3), `ring_shared_phone`
(phone 72h>=3). Gross precision all block-tier (never-paid 92.2% / 88.0% / 97.5%,
~21-23x). **Unique (marginal) capture differs sharply: device unique_n=232 @ 46%
(the real net-new contributor); persistent unique_n=26 @ 24% (small volume but the
#6988 virtual-number antidote — strategic); phone unique_n=2 @ 0% (essentially
fully redundant — every phone-ring row already matches a device/identity/account
ring). Phone earns its place as typology documentation, not marginal capture —
flagged for wendao to keep-or-drop.** Register union now captures 4,597 rows @
87.1% never-paid; LOW-band discovery (heuristic-missed) 425 @ 84.9% (vs ~2 in
round-2). Residual never-paid 3.82%.

**Next-layer discovery on the POST-LOCK residual (`analysis/residual_next_layer.py`,
target DPD45, mature-only, train->test validated):** the strongest remaining bucket
has NO sharing edge — **neobank x fresh-account x small-amount x low-velocity**:
`is_neobank AND plaid_acct<=37.5d AND avg_prior_advances/day<=0.089 AND
total_disbursed<=$29.49` -> DPD45 33.5% (5.8x), never-paid 30.2% (6.0x), n~1013
test. Notably **fraud-smelling: ~90% of its DPD45 never repaid** (not late-repaid
credit stress). But with no edge and ~30% precision it is a REVIEW/MITIGATE queue
(step-up verification / lower first-advance limit on fresh neobank small advances),
NOT a block gate — ~70% would be false positives if blocked. Same fast-small-dollar
-churn cohort round-3 found, sharpened by the neobank flag. (Caveat: two leaves
split on prior_min_hours~247 = NaN-imputed median, an artifact.)

**Window note (answers wendao):** 72h/7d/30d ALREADY exist for every new edge — no
rerun needed; precision DECAYS with the window (device 30d>=2=32% vs 72h>=3=80%),
so the locks use the SHORT windows on purpose. Extending the new edges beyond 30d
(90d/lifetime) is the only window change that needs a SQL addition + overnight
rebuild.

**Follow-ups (2026-06-08, wendao review):**
- **`ring_shared_phone` DROPPED** (unique_n=2 @ 0% — fully redundant with the
  device/identity/account rings). Register now 4 scenarios, version 2026-06-08.2.
  Kept a note in register.yaml recording the decision.
- **Going narrow does NOT lift the no-edge cohort past ~30%.** Re-ran the
  next-layer tree at depth 6 / min_leaf 60, sorted by never-paid: the purest
  pocket is still `first-advance AND neobank AND total_disbursed<=$29.49` ->
  29.7% never-paid (5.9x, n=992), and every finer cut hovers 20-33%. **~6x is a
  real CEILING for the no-sharing-edge cohort — there is no hidden block-tier
  pocket inside it.** The sharing EDGE is what buys block-tier precision (80-98%);
  behavioral/amount/velocity features without an edge top out at review-tier.
  So this cohort is a mitigate/review queue, full stop (confirmed, not a tuning
  artifact). `analysis/residual_next_layer.py` now takes --max-depth/--min-leaf/
  --sort for this drill-down.
- **IP verdict (`analysis/ip_screen.py`): raw IP sharing is DEAD, even at the
  leaky current-state ceiling** — ip_address users>=2 -> 4.7% (0.9x), >=10 ->
  4.4% (0.9x); signup_ip ~1.0x. NAT/households fully swamp it (confirms round-1).
  Do NOT build an as-of users_on_ip edge. Two faint extras: signup_ip==latest_ip
  -> 8.4% (1.7x, weak modifier), has_ip_address==0 -> 11.8% (2.3x, n=1950,
  confounded). **The fraud-shaped IP signals are DERIVED, not raw, and need an
  IP-intelligence enrichment (Tier-3 pull): datacenter/hosting/VPN/proxy flag
  (real borrowers don't advance from AWS/a VPN) and IP-geo vs KYC-address/phone
  -area-code mismatch.** Parked in TODO Tier-3.
- **Institution concentration (`analysis/institution_screen.py`): the institution
  signal IS Chime.** Chime Bank = 71,284 residual-mature rows @ 11.4% never-paid
  (2.3x), 100% neobank — i.e. `is_neobank_high_risk` is essentially Chime. No
  institution exceeds 11.4%, none reaches 15%, and Chase is not a concentration.
  Institution adds nothing sharper than the neobank flag. The shell-vs-real
  discriminator that WOULD lift this cohort is **income/payroll presence** (Plaid
  txn pull, Tier-3) — separating a bust-out shell account from a genuine new
  Chime user who defaulted.
- **"Proven algorithm" for permutations: subgroup discovery via beam search**
  (`analysis/subgroup_discovery.py`, self-contained — pysubgroup not installed).
  Enumerates selectors, beam-searches conjunctions, ranks by a quality measure;
  rigor = discover-on-train / validate-on-held-out-test + a binomial significance
  p-value + candidates-evaluated for Bonferroni. It found a combination the
  greedy tree MISSED: `is_neobank AND prior_advances<=2 AND
  signup_ip_matches_latest_ip==1` -> 20.5% never-paid (4.1x), n=786 test,
  p=1.3e-87 (signup_ip_match alone was only 1.7x; it combines). Largest robust
  pocket: `plaid_acct<=86d AND neobank AND prior_advances<=2` -> 19.1% (3.8x),
  n=6505 test, p~0. **But the CEILING is now confirmed three independent ways
  (greedy tree, fine tree, exhaustive beam search w/ significance): nothing in
  the post-lock residual exceeds ~28% never-paid, and the only >25% pockets are
  tiny device==2 sub-threshold remnants. There is NO block-tier conjunction
  left.** The residual is exhausted for block rules; lifting it needs NEW
  discriminating features (income/payroll), not a better search algorithm. The
  best the residual offers is review-tier (~4x, neobank x early-tenure).

## 2026-06-08 — Tier-1 feature rebuild; team monitoring SQL is forensic, ours is predictive

**Cross-checked our pipeline against the team's fraud apparatus** (Incidents
#6848 & #6988 dashboard, the 3 Tabapay clawback queries, the combo-detail pull,
and `scoring_model_20260429.sql`). The central distinction: **the team's sharing
counts are current-state / hindsight** (`base_prod__plaid_accounts_current_state`,
unbounded `COUNT(DISTINCT user_id)`) — correct for *clawback/recovery* but leaky
as a *predictive* feature; **ours stays as-of**. Their ~$1M→$2M is a hindsight
loss figure, not a prediction target, and their scored impact spans Likely +
Extremely-Likely, not just EXTREMELY_LIKELY. **#6988 (Chase virtual account
numbers) defeats account-number-keyed sharing on BOTH sides** (each virtual number
is a distinct `account_number`, so the count never accumulates) — parked, we don't
have the virtual-card data.

**Built the Tier-1 feature set** (full plan: TODO.md ⭐ CONSOLIDATED FEATURE-ADD
PLAN). Inlined the upstream into `base_table.sql` (out-of-band DDL retired →
`data/queries/archive/`), added as-of features: scarce-resource sharing edges
(device / persistent-id / address / phone / email, mirroring
`users_on_bank_account_*`), Jaro-Winkler name-match, `is_neobank_high_risk_institution`,
`is_joint`, prior-only advance velocity (`prior_min_hours_between_advances`), and
the team's detection flags (3-in-72h / 5-ever / **10-ever**). New table
`fraud_advance_feature_base_automl_v2`, dataset **`v1_76d3ad45`** (1,021,950 rows ×
113 cols; old `_automl` table left intact; experiment still `fraud_anomaly_v1`).
`heuristic_fraud_score`/`band` kept **byte-identical** (proxy-label comparability).

**Due diligence before the single materialize** (no repeated builds): live fan-out
scan showed **no many-to-many blowup** (email ~unique, phone max 427, address 81,
persistent 576) EXCEPT device — `user_client_metadata` is SCD (176M rows,
~11/user-device) → deduped `device_links`. EXPLAIN compiled end-to-end; adversarial
review clean on as-of/grain/dedup.

**Validation:** splits healthy (positives in train AND test); all sharing edges
populated; name-match carries signal (**last-name mismatch ~1.25× never-paid lift**,
independent of the rings).

**Lesson (cost us a noise column):** `official_name` is the account **product
type** (Checking / Varo Checking / Individual Account), **not** the holder name —
`name_match_official` was noise and was dropped. *Scan column CONTENTS, not just
existence, before trusting a derived feature.* Real holder-name match needs the
Plaid identity/owner source (Tier-3 pull). `v1_76d3ad45` still carries the dropped
noise column; remove on the next rebuild.

## 2026-06-07/08 — unsupervised sweep closed; supervised lens finds a heuristic-missed cluster

**Primary metric moved to gross-DPD45 AP** (from never-paid). On the pinned
snapshot the two are 89% the same: never-paid ⊂ DPD45, and the 11% difference
(delinquent-but-cured) sits in LOW/POSSIBLE — the fraud bands have zero cured
cases. Picked DPD45 as the team's standard ruler; never-paid kept as a
secondary. The choice only moves ~490 rows and is invisible inside the fraud
bands.

**The unsupervised anomaly sweep is closed and negative on the residual.**
Three geometries on the scenario-gated residual, scored on residual gross-DPD45
AP (base 0.057): Isolation Forest (axis-aligned) **0.075**, GMM (density)
**0.069**, MLP autoencoder (reconstruction) **0.055** — the autoencoder joins
round-1's linear PCA as a reconstruction approach that underperforms. All three
sit at ~base rate and merely re-rank the heuristic's own POSSIBLE/LIKELY
velocity band (mean percentile 93–99); the LOW discovery-target band stays at
the median (~49). **In the gated residual, "anomalous" and "DPD45/fraud" are
not the same direction** — the anomaly-for-discovery premise held for the
(now rule-handled) ring but breaks for what's left.

**A supervised model is the instrument that works.** A HistGBM ceiling probe on
the same gated residual reaches **0.16 AP / 5.9× lift at top-1%** — ~2× any
unsupervised view — leaning on a velocity × amount × account-newness
interaction. The residual signal is label-aware interaction structure in the
bulk of the distribution, not extremeness, so only supervised selection
recovers it.

**Supervised-lens discovery (the payoff).** The GBM's top-1% residual rows are
**90% LOW band (heuristic-missed) at 29% never-paid (~6× base)**. Profile: high
advance velocity (~6× base `avg_prior_advances_per_day`), *small* amounts
(~half base `total_disbursed`/`loan_amount`), very new bank account (days vs
months `days_since_plaid_account_created`) — a small-dollar fast-cycling-on-
fresh-accounts pattern, distinct from the registered amount>$100 ring
scenarios. Moderate and intent-ambiguous (fast small-amount churn is also
credit stress), so review/POSSIBLE-tier, not the 89–96% block rings; needs case
review to separate mule-cycling from credit-churn. Reusable tooling, both run
from the pinned snapshot by id: `analysis/ceiling_probe.py` (supervised ceiling
+ feature attribution) and `analysis/supervised_lens.py` (top-risk cohort
characterisation).

**The next axis, honestly — device / persistent-id, as-of (CORRECTS the leaky
screen).** The 2026-06-06/07 entry and `TODO.md` quote device sharing at
"≥3 users → 81.6% never-paid on 69 register-invisible rows." That was a
*whole-snapshot* count — leakage-inflated (the first advance on a shared device
"saw" future users). Re-run **as-of, prior-only, windowed** (the
`users_on_bank_account_*` convention), the honest numbers on the residual-mature
population (base 5.1%) are:
- `device_id` ≥2 distinct users within **7d**: 34 rows @ **79%** (15.6×); 72h is
  tighter (23 @ 87%). Precision *decays* with the window (lifetime → 28%
  marginal), so there is innocent stale device reuse → short window.
- `persistent_account_id` ≥2 within 7d: 7 rows @ **100%** (20×). Precision holds
  at 100% at *every* window (no innocent stale reuse — same real Plaid account,
  two identities), but lifetime adds only 1 row over 7d, so 7d for compute.
- De-duped net-new union ≈ **35 rows @ 80% (+28 never-paid), ≈ +9%** on top of
  the registered rings' ~375 / 334. Block-tier precision, modest volume.

So unsupervised is closed; the path is **supervised features + precise rules on
new edges**. Next build = device + persistent-id as-of features + two scenarios
(full detail in `TODO.md`); Track B = neobank feature, ACH codes, deposit
history, name-match — parked. New convention: every scenario's `theory` records
its shape/abnormality stat, no-innocent-version argument, and window rationale.
Screening tooling (the pre-materialization screens `current_data_screen.py`,
`asof_sharing_screen.py`, `asof_breakdown.py` — REMOVED in the 2026-06-08 cleanup
once the edges were materialized; now done properly on the live columns by
`analysis/edge_precision_screen.py`).

**Process.** Two round-4 launches no-op'd silently (exit 0, no trial) because
the loop's context-render step shell-evaluates `--instruction`; metacharacters
(`->`, `()`) in the text broke it. Fix: keep instruction text
metacharacter-free. Candidate library to-do — render-context should not
re-evaluate `--arguments` through a shell.

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
