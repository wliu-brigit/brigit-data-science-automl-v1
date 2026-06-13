# Fraud control system — design spec (DRAFT, not signed off)

Design for the holistic discovery → plug → monitor system, converged in the
2026-06-13 brainstorming session (wendao + Claude). Built fresh in `codex_poc/`
(today's "Neo4j POC" grows into it; rename later). Sign-off: pending. Thresholds
and which keys/methods actually work are deferred to the v3 data pass — this
spec fixes the *shape*, not the tuned numbers.

Answers to the guiding principles in [`../PRINCIPLES.md`](../PRINCIPLES.md);
the graph schema it discovers over is in
[`SCHEMA_DESIGN.md`](SCHEMA_DESIGN.md).

> **Convention departure (capture in the repo README too):** this system does
> NOT use MLflow as its durable record. MLflow is the AutoML package's
> experiment store; this is operational fraud-control state, inherently
> different. Durable state lives in warehouse / DuckDB / GCS.

## 1. Purpose & the two-layer model

The system is a control loop (PRINCIPLES P1). Its load-bearing distinction:
**discovery and plug-the-hole are two layers — different logic, different
validation.**

- **Discovery** — "where is the fraud?" Precise, but often NOT deployable
  (production can't run the graph query or the expensive as-of join).
  Validated against DPD45 alone.
- **Plug-the-hole** — "what can we cheaply block?" The deployable *projection*:
  a shared, production-checkable key (bank account / device / phone / address /
  persistent account / IP). Blunter than the discovery that motivated it, so it
  has two error modes that are BOTH measured: over-reach (blocks innocents who
  merely share the key) and under-cover (misses discovered fraud that shares
  nothing cheaply blockable). Validated against DPD45 **and** discovery-coverage.

We produce and validate; **production consumes** the plug list. How they
enforce it (blocklist lookup, joined per-entity flag) is their integration.

## 2. Architecture

Four parts, each its own mechanism, tied by one daily snapshot pipeline. The
**finding store is the single seam** — discovery writes it, plug-derivation
reads it, monitoring measures over it; nothing else couples the layers.

```
 daily snapshot                ┌─ DISCOVERY REGISTRY ── "where is the fraud?"
 warehouse → DuckDB            │    two separate method kinds, one output contract:
 → Neo4j mirror               │      • scenario methods  — declarative predicates (consume register.yaml)
        │                      │      • graph methods     — Neo4j/GDS: communities, components, PPR, dense
        │                      │    each validated independently vs DPD45
        ▼                      ▼
   run all methods ───────►  FINDING STORE ── durable record:
                              │   flagged user/entity × method × snapshot × outcome
                              ▼
                          PLUG REGISTRY ── "what can we cheaply block?"
                              │   extract → validate → qualify (§5); sticky lifecycle (§4)
                              ▼
                          [ production consumes the burned-key list ]
                              │
                          MONITORING SPINE ── §7: discovery rate · prevention ·
                              leakage · precision drift → drives plug expiry + flywheel
```

Scenarios are **consumed, not reimplemented — but the integration is ours to
design.** The scenario *definitions* in `scenarios/register.yaml` stay canonical
(they also feed the ML gate, so forking them would create two truths), but the
new system is free to adapt them into its own discovery-method interface rather
than wrap the existing `engine.py` as-is — whatever fits the holistic system
more cohesively. Everything else is built fresh.

**Primary design goal — an extensible end-to-end skeleton.** The near-term
deliverable is the *whole pipeline in skeleton form* behind a **stable
discovery-method contract**: any discovery method (a scenario, a graph
algorithm, a future GNN lens) emits findings in one standard shape and plugs
into the finding store without touching plug-derivation, monitoring, or
persistence. We build the skeleton *now*, before v3 tuning, precisely so the
many discovery methods still to come slot in rather than force a rebuild —
think the problem through end-to-end once, plug in forever after.

## 3. Leak-free evaluation — two states, not a sliding window

Ring discovery runs on as-of **state**, not a per-event time window: a user who
looks innocent early is correctly pulled into a ring once it forms later,
because discovery reads the full accumulated state at the moment it runs (by
design). So the holdout is exactly two states:

- **State A** = everything before the held-out month.
- **State B** = everything including it (full state).

Derive + plug on **A**; measure realized effect against what's new in the A→B
delta. One guardrail keeps it honest: A's *derivation* uses only outcomes
already mature as-of A's cutoff (no plug built from a label that hadn't happened
yet). Ring *membership* legitimately uses full as-of state. Evaluation uses the
month's matured outcomes — that's the point of holding it out. This doubles as
a dress rehearsal for the eventual live loop: swap the held-out month for
"yesterday" once production deploys and feeds back.

## 4. Persistence model

Split by what changes how often:

**Definitions** (slow, reviewed, version-controlled, in-repo): the
discovery-method catalog (scenarios = `register.yaml`; graph methods = which
GDS/Neo4j queries are "live" + params) and plug-derivation rules. The single
tunable config (§6) lives here too. Editing one is a deliberate reviewed act.

**Accumulated data** (regenerated each run, grows over time) — and findings vs
plugs persist *differently*:

- **Findings = historized snapshots (log-like).** Each run writes a dated
  snapshot keyed by **refresh** (run id/date) + logic-version + data-version.
  "Current" is the latest regeneration; history is kept. A *shrink is
  diagnosable*: diff two snapshots and the version tags say whether the data
  moved (entities aged out) or the logic changed. Trim to deltas / skip when
  nothing material changed — keep the trail lean. This history is what the
  monitoring spine compares run over run.
- **Plugs = accumulating, lifecycle-managed registry (sticky).** Once a key is
  burned and production acts on it, it must NOT disappear because a later
  rebuild didn't re-derive it (that would un-block a real mule account). A plug
  has an explicit lifecycle — *added* (newly derived AND validated) → *active* →
  *expired* — and leaves "active" only by a **deliberate** rule (gone cold, aged
  out, caught innocents on review), never by "didn't reproduce today." A shrink
  in supporting findings is a signal to *review*, not auto-remove.

**Storage homes:** definitions → in-repo (versioned). Finding store + monitoring
history → DuckDB/parquet (rebuildable, matches the graph-store pattern), durable
copy in GCS if it must outlive the laptop. Production-facing burned-key list →
warehouse (Snowflake) table = the consumption contract (early iteration local
parquet; warehouse the deploy target). Not MLflow.

## 5. Plug derivation — extract → validate → qualify

One derivation pass per refresh, with a deliberate seam before qualification:

1. **Extract** (mechanical): from the finding set, enumerate shared candidate
   keys and their support — "entity E, touched by k discovered-fraud users."
2. **Validate** (measurement, leak-free on State A): per candidate, compute the
   stat panel → precision vs DPD45 (over-reach / innocent-capture check),
   coverage vs discovery (under-cover check), volume, explicit innocent count,
   corroborating key-type count. Output = a **persisted candidate-stats table
   (facts)**.
3. **Qualify** (decision, cheap filter over the facts): a candidate becomes a
   plug only if it clears conjunctive gates — precision ≥ τ (block-tier ~80%)
   AND support ≥ m AND coverage contribution ≥ c. Gates, not an additive score
   (project's anti-additive stance). Thresholds are parameters (§6).

Because extract + validate are expensive (touch the graph/data) and their output
is persisted, **retuning a threshold re-runs only the cheap qualify filter** —
never the derivation. Key choice favors **maximum coverage at minimum innocent
capture** (the literal "plug the right hole"); **multi-key corroboration** (burn
a user only where they share ≥2 independent key types — the `SHARES_RESOURCES`
lever) is the knob that pushes precision up and keeps families / coincidental
sharers out.

## 6. Tunable parameters — one config, never baked

All thresholds live in a single in-repo config, applied as the §5 qualify filter
and §4 expiry rules over persisted facts (P6). Initial set: block-tier precision
τ, min support m, min coverage contribution c, corroboration key-type count,
plug expiry rules (cold/age/precision-drop), holdout window length. Changing one
is a one-line, version-controlled, instant-to-rerun edit; it never triggers
re-derivation.

## 7. Monitoring spine

A daily job (run as the §3 holdout replay until production feeds back) spanning
both layers, reporting per refresh: **discovery rate** (new rows matching a
scenario/graph method — is fraud still flowing in?), **prevention rate** (new
bad caught by active plugs), **leakage** (discovered fraud no plug caught → the
next holes), **precision drift** per active plug (drives expiry), **residual
shift**. Feeds the flywheel: leakage → new findings → new plugs; drift → expiry.
Monitoring is what keeps the accumulating plug list honest.

## 8. Build staging & dependencies

Design-now, build-staged behind data:
- **Needs v3 data** (the real discovery output): thresholds, which keys clear
  the bar, which graph methods earn cataloging, `SHARES_RESOURCES` size.
- **Needs VPN/prod**: warehouse read for the v3 store rebuild; warehouse write
  for the production-facing plug table; the link-grain + IP-key SQL.
- **The skeleton — build now (the priority).** A *walking* skeleton: the
  end-to-end pipeline with the stable discovery-method contract + finding-store
  + snapshot/trim machinery, the extract→validate→qualify shape with the
  cheap-filter seam, the two-state holdout harness, the config scaffold — **plus
  a representative few real discovery methods wired through the contract**
  (≥1 scenario adapted from the register, ≥1 graph method from the
  `analysis/`/`codex_poc` runners), deliberately **not exhaustive**. Enough to
  prove the loop runs on real findings and that new methods plug in — you can't
  validate a plug-in contract with zero plug-ins. All on the sample; v3 is
  needed only later for tuning and the full method set. Built once, the
  skeleton is what lets new methods and v3 tuning plug in rather than trigger a
  rebuild — so it's worth building before the tuning data exists.
- **Depends on** the `SCHEMA_DESIGN.md` edge model (itself design-only) for the
  graph methods + corroboration keys.

## 9. Open items

- Finding-store record schema (exact columns of the shared discovery output
  contract) — detail at build time.
- Graph-method catalog format (how a Neo4j/GDS method is registered "live" +
  params) — lighter than the scenario register; shape TBD.
- Plug expiry policy specifics (cold/age/precision thresholds) — tune on v3.
- Whether the production-facing list is value-keyed only or also carries
  precomputed per-entity flags — depends on the production consumption seam
  (currently unknown; flagged as production's side).
