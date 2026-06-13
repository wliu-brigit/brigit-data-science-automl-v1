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
