# Fraud graph & detection — guiding principles (DRAFT — not signed off)

**Why this file exists:** these principles are the shared understanding behind
the graph detection effort — what the system is for and the rules every design
decision answers to. If you are building on top of this work and your change
would violate one of these, stop and raise it before building — that friction
is the point: it is how drift gets caught early. Principles change rarely and
loudly: amend in a dedicated change with the reasoning recorded.

Status: draft from the 2026-06-12 schema discussions (wendao + Claude);
sign-off pending (wendao). Schema-level working notes live in
[`codex_poc/SCHEMA_DESIGN.md`](codex_poc/SCHEMA_DESIGN.md).

## P1 — The system is a control loop; the graph is its discovery instrument

```text
DISCOVER   full-history snapshot graph + backdated outcomes
           → find structure that concentrates bad outcomes
DISTILL    convert each finding into the cheapest enforceable artifact
VALIDATE   patterns: leak-free as-of measurement on history;
           entity blocks: the ring evidence behind the value
ENFORCE    real-time, cheap checks only — no graph at decision time
MONITOR    per control: hits, precision over time, residual shift;
           feeds the next DISCOVER pass (the flywheel)
```

DISTILL produces two control types with different machinery — a **scenario**
(pattern gate: a predicate over as-of features; generalizes; needs the feature
computable at decision time) and an **entity block** (plug-the-hole: a
specific burned value; surgical; O(1) lookup) — plus a softer third tier, the
**review queue** (not confident enough to block; a human looks).

Findings are worth more the lower their enforcement cost: entity block >
feature rule > community membership (which is reducible to an entity block —
you join a ring by touching its entities).

Label states: **known-bad** (mature, outcome observed) validates controls
backward; **presumed-bad** (new account matching a validated control) is what
makes proactive blocking legitimate; **unknown** (immature, unmatched) is the
residual where discovery hunts next.

Responsibility split: Neo4j owns DISCOVER and review display. DuckDB/warehouse
own grain truth, time, validation, the durable registry, and monitoring
history. The mirror is rebuilt daily and disposable — it displays state, it
never owns state.

## P2 — Discovery is rich; enforcement is cheap

Expensive graph work happens offline to find things; what ships forward is
always a cheap artifact. No contradiction between rich graph features in
discovery and cheap real-time controls — the graph never runs at decision
time. Judge schema choices by whether they sharpen discovery and keep
findings reducible to cheap controls.

## P3 — Lossless store, opinionated views

Applies to the DuckDB store AND the Neo4j mirror. Information is excluded at
analysis/projection time, never at ingest. (Worked example: IP — admitted as
information despite failing as a feature; excluded from default projections,
not from the graph.)

## P4 — Time lives in DuckDB

Scenario windows are as-of anchored per advance; a naive "last 72h" over edge
timestamps matches them on only **4.4% of rows** (measured,
`analysis/graph_store_crosscheck.py`). Scenario computation never moves into
Neo4j; the mirror carries scenario *results* and time *summaries* sized to
what discovery needs.

## P5 — Outcomes are advance-grain; the fraud unit is the identity

Our fraud is first-party: the person never intended to pay and advances are
extraction events — so the fraud unit (and the graph node) is the user, and
advances are evidence *about* the user. (Same split industry makes: payment
fraud — good customer, stolen card — puts transactions in the graph;
identity/ring fraud collapses to the identity.) Every user-level number is a
derived roll-up: each roll-up is an explicit recorded rule, and counts are
carried so rates and stricter definitions stay computable. Immature advances
count toward nothing — unknown is not good. The advance grain itself stays in
DuckDB.

## P6 — Thresholds are parameters, never baked into poured facts

Every outcome question gets dual semantics: a **sensitive evidence marker**
(ever-bad — one never-repaid advance is real ring evidence; catches the
bust-out shape) and a **strict definition** as a query-time threshold over a
poured rate (default 0.8, adjustable without re-import) — used where
credit-stress pollution hurts: concentration metrics, queue ranking. Artifacts
that apply a threshold self-document which one produced them.

## P7 — Discovery looks back on full state; deployment is leak-free

Discovery runs on the **full accumulated as-of state**, not point-in-time at
each decision: a user who looked innocent early is correctly pulled into a ring
once it forms later. Hindsight is legitimate for *finding* fraud — the job is
to learn where it is. But every control we **promote to deployment** is
re-validated **leak-free** via a two-state holdout — derive on the state before
a held-out period (the holdout window; ~1 month to start, tunable), measure on
the new activity in it — so a precision or coverage claim never uses an outcome
the derivation could not have known.
**Discovery may look back; promotion may not.** (This is distinct from P4: the
*scenario windows* are as-of anchored per advance; this principle governs
graph/ring *discovery* and the validation boundary. The project has been burned
by leakage-inflated screens before — that is why the line sits at promotion.)

## P8 — This is an operational system, not an AutoML experiment

The fraud-control system's durable state — discovery findings, the burned-key
plug list, monitoring history — is **operational**, and lives in
warehouse / DuckDB / GCS. It does **not** use MLflow: MLflow is the AutoML
package's experiment record, and this system is inherently different. (Slow,
reviewed definitions stay version-controlled in-repo; thresholds are
parameters, P6.)

## P9 — Discovery and enforcement are separate mechanisms, validated differently

P1's two control types are kept structurally separate. **Finding** fraud
(scenarios, graph methods) and **blocking** it cheaply (a plug — a shared,
production-checkable key) are different mechanisms with different validation:

- discovery is validated against outcomes (DPD45);
- a plug is validated **twice** — against outcomes (precision: would it block
  innocents?) *and* against the discovery set (coverage: does it catch the
  fraud we found?) — because the deployable projection is blunter than the
  discovery that motivated it.

Monitoring is the ongoing flywheel: it re-validates active plugs, drives their
expiry, and turns leakage (discovered fraud no plug caught) into the next holes
to plug. We produce and validate; production consumes.
