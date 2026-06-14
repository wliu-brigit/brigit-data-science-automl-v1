# Fraud Control End-State Design

Date: 2026-06-14

Status: draft for review

## Purpose

This spec reframes the fraud-control work around the desired end state instead
of the current file-by-file implementation. It is meant to guide code structure,
workflow, and gap analysis for the fraud graph and detection system described in
`projects/fraud_anomaly_detection/PRINCIPLES.md`.

The goal is an operational fraud-control workbench:

1. Register discovery methods.
2. Run them on a snapshot.
3. Store what they found with evidence and versioning.
4. Select findings worth promotion.
5. Derive cheap production-enforceable plugs.
6. Validate discovery and plugs without leakage.
7. Monitor active controls and feed residual leakage back into discovery.

## North Star

The system is a control loop, not an AutoML experiment and not a graph-script
collection. Rich graph and scenario logic can be used offline to discover fraud,
but anything production consumes must reduce to a cheap, explicit control:

- a scenario rule over as-of features,
- an entity plug over a shared key,
- or a review queue for evidence that is not block-tier.

Discovery and enforcement must remain separate. A discovery method can be good
at finding fraud and still be bad for production enforcement. A plug can be easy
to enforce and still be too blunt. The system must measure both layers.

## End-State Concepts

### Discovery Method

A method finds suspicious users, entities, or communities. Scenarios, graph
methods, subgroup rules, and future model/GNN lenses are all discovery methods.
They share one output contract but carry metadata that makes their semantics
clear.

Each method should declare:

- `name`
- `version`
- `method_type`: `scenario`, `graph`, `model`, or `subgroup`
- `time_semantics`: `snapshot_review`, `leakfree_asof`, or `production_safe`
- `promotion_tier`: `evidence_only`, `review_queue`, or `plug_candidate`
- `enforcement_projection`: `entity_key`, `scenario_rule`, or `none`
- parameters needed to reproduce the method

This prevents an exploratory graph queue from being accidentally treated as a
production-ready block rule.

### Finding Store

The finding store is the durable seam between discovery and the rest of the
system. Discovery writes to it. Selection, plug derivation, validation, and
monitoring read from it.

Findings are snapshot history, not sticky state. Each snapshot records:

- refresh id or run date,
- data version,
- method name and method version,
- subject user/entity,
- score,
- evidence,
- method metadata,
- content hash.

Empty snapshots should still persist. A method that finds nothing is a real
observation.

### Selection

Selection decides which findings are worth promoting from evidence into action.
It dedupes across scenarios and graph methods, measures marginal contribution,
and blocks low-precision or duplicate graph methods from driving plugs.

Selection should be reusable core logic, not report-only logic.

Typical selection facts:

- total users found,
- overlap with baseline scenarios,
- net-new users beyond current selected set,
- marginal DPD45 user and advance rate,
- support,
- stability across backtest windows,
- method class and promotion tier.

### Plug System

The plug system converts selected findings into cheap controls. The main plug
type is a burned entity key such as device, bank account, persistent account,
phone, address, email, or IP.

Plug derivation has three separate steps:

1. Extract candidate keys touched by selected discovery users.
2. Validate candidate facts on the derivation state.
3. Qualify plugs by threshold policy.

Candidate facts are persisted so threshold retuning can rerun qualification
without redoing expensive extraction and validation.

Plugs are sticky lifecycle state, not snapshots. A plug should not disappear
only because a later discovery refresh did not reproduce it.

Lifecycle states:

- `proposed`: candidate cleared current gates but is not active.
- `active`: production-facing or ready for production consumption.
- `expired`: deliberately removed by age, coldness, or precision decay.
- `rejected`: reviewed and intentionally not promoted.

### Validation

Validation has separate jobs for discovery and plugs.

Discovery validation asks: did the method find users with bad outcomes?

Plug validation asks two questions:

- precision: would this plug catch bad users or block innocents?
- coverage: does this plug catch the fraud discovery found?

Plug reports should preserve the operator buckets:

- `covered_discovery`: discovered users the plug catches.
- `uncovered_discovery`: discovered users with no deployable plug coverage.
- `outside_discovery`: users touched by the plug but not in discovery.

`outside_discovery` is not automatically innocent. Its outcome rate tells us
whether the plug overreaches or finds extra bad users.

### Backtesting

Backtesting should be first-class, not a side package. The default promotion
test is a two-state replay:

- State A: derive methods, findings, candidate facts, and plugs.
- Holdout delta: measure what those plugs would have caught later.

Monthly or rolling historical backtests are the stability lens for promotion
and expiry. They should reuse the same discovery, selection, plug, and
validation modules as the daily runner.

### Monitoring

Monitoring keeps active plugs honest after promotion. It reports:

- active plug hit volume,
- prevention rate,
- precision drift,
- innocents blocked,
- leaked bad users,
- uncovered discovery,
- outside-discovery outcome rate,
- residual shift.

Monitoring should feed two queues:

- expiry/review candidates for plugs that decay or overreach,
- new discovery work for leaked fraud that no plug catches.

## Target Code Shape

The final code should be organized by system responsibility:

```text
control/
  methods/
    base.py
    scenario.py
    graph.py
    catalog.py

  findings/
    contract.py
    store.py
    selection.py

  validation/
    outcomes.py
    split.py
    discovery.py
    plugs.py
    backtest.py

  plugs/
    candidates.py
    qualify.py
    registry.py
    export.py

  monitoring/
    daily.py
    leakage.py
    drift.py

  reports/
    operator.py
    json.py

  run.py
  config.py
```

Each folder answers one operator or developer question:

- `methods`: how do we find fraud?
- `findings`: what did we find and why?
- `validation`: did it really work?
- `plugs`: what can production cheaply enforce?
- `monitoring`: is it still working?
- `reports`: how do humans and machines consume the result?

## Ideal Run Workflow

The daily runner and the backtest runner should execute the same conceptual
pipeline:

1. Load the snapshot store.
2. Load enabled discovery methods from the method registry.
3. Run methods and produce `FindingSet`s.
4. Write finding snapshots.
5. Validate each method and the deduped discovery union.
6. Select promoted discovery findings.
7. Extract candidate plug keys from selected findings.
8. Validate plug candidates on State A.
9. Qualify plugs with the configured policy.
10. Evaluate qualified plugs on holdout.
11. Update the sticky plug registry.
12. Export active plugs.
13. Report coverage, leakage, outside-discovery, drift, and next actions.

## Extensibility UX

The system should be easy to operate without understanding internal graph code.
Adding or removing a control surface should be a registry or config action, not
a multi-file edit.

Expected extension paths:

- Add a scenario: update `scenarios/register.yaml`, then enable the scenario
  method in the method registry.
- Add a graph method: implement one adapter that emits `FindingSet`, then add
  one registry entry with metadata and parameters.
- Disable a method: flip one registry entry to disabled or remove it from the
  enabled profile.
- Tune plug qualification: edit threshold config and rerun qualification over
  persisted candidate facts.
- Remove a plug: transition it through lifecycle state, with a recorded reason;
  never delete it by omission from a later derivation run.

This should eventually support named profiles such as `review`, `candidate`,
and `production`, so a user can run broad discovery without accidentally
exporting every discovery method into enforcement.

## Lessons From Current Implementations

### Keep From The Reference Repo

The reference `fraud-anomaly-detection` project is stronger as a discovery lab.
It has broader graph ideas, backtest-panel thinking, and useful historical
replay concepts.

Keep:

- broad graph discovery methods,
- review queues,
- monthly and historical backtest ideas,
- graph queues as investigation surfaces,
- the instinct to search residual fraud after scenarios.

Do not copy directly:

- hard-coded control wiring,
- weak separation between exploratory graph evidence and production action,
- plug derivation that can validate too broad a key space,
- report logic that owns too much control behavior.

### Keep From The Current Skeleton

The current skeleton is stronger as an operational control loop.

Keep:

- common `FindingSet` output shape,
- method versioning and finding snapshots,
- discovery-vs-plug validation separation,
- `covered_discovery`, `uncovered_discovery`, and `outside_discovery`,
- State A / holdout framing,
- report persistence,
- scoped plug candidate validation.

Fix:

- `run_skeleton()` and `selected_discovery_report.py` are not one coherent
  system yet.
- Graph selection logic lives too much inside a report module.
- The method catalog is too thin and lacks method semantics.
- The plug registry is report-level, not lifecycle-managed operational state.
- Backtesting is not a first-class reusable module.

## Gap Map

### Gap 1: Unified Method Registry

Current state: one small catalog drives `run_skeleton()`, while the richer
selected report has its own graph and scenario logic.

Target: one registry drives daily runs, selected reports, and backtests. Each
method declares type, time semantics, promotion tier, parameters, and version.

### Gap 2: Method Classification

Current state: graph methods can be treated as methods without explicit safety
class.

Target: every method is visibly classified as snapshot evidence, leak-free
as-of feature/rule, review queue, or plug candidate.

### Gap 3: Reusable Selection Layer

Current state: marginal graph selection exists primarily in the selected report.

Target: selection is a reusable module that reports total, overlap, net-new,
marginal rate, and promotion eligibility.

### Gap 4: Sticky Plug Registry

Current state: burned keys are derived into reports.

Target: plugs persist as lifecycle-managed operational state with proposed,
active, expired, and rejected statuses.

### Gap 5: First-Class Backtesting

Current state: backtest thinking exists, but is split between historical
reference code and current report flow.

Target: backtests reuse the same runner modules over historical State A /
holdout slices and produce stability evidence for promotion and expiry.

### Gap 6: Operator UX

Current state: a user needs to understand too many files.

Target: the normal actions are obvious:

```bash
fraud-control methods list
fraud-control run --store ...
fraud-control report latest
fraud-control plugs qualify --threshold-profile ...
fraud-control plugs export-active
fraud-control backtest monthly --store ...
```

The code may begin as Python module entry points, but the workflow should be
designed around these operator actions.

## Non-Goals

This design does not tune thresholds, choose final production graph methods, or
decide the Snowflake production table contract. Those need v3/full data and
production integration details.

This design also does not require graph execution at decision time. Enforcement
must remain cheap.

## Success Criteria

The design is successful when:

- adding a scenario means updating the scenario register and enabling one method
  entry;
- adding a graph method means adding one adapter and one registry entry;
- reports no longer contain core selection or plug logic;
- every active plug can explain which selected findings created it;
- every plug has precision, coverage, outside-discovery, and holdout evidence;
- a plug can expire only through an explicit lifecycle rule;
- backtests and daily runs use the same core modules;
- an operator can understand the workflow without reading internal graph code.
