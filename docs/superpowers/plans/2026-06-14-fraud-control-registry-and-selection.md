# Fraud Control Registry And Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first end-state foundation: explicit discovery-method metadata, a reusable graph/scenario selection layer, and a selected-discovery report that consumes shared selection logic instead of owning it.

**Architecture:** Keep the current `codex_poc/control` layout for this phase and add focused modules rather than doing a broad directory move. Discovery methods will expose metadata, the catalog remains the reviewed extension point, and graph-method marginal selection moves into a reusable module that both reports and future runners can consume. This plan intentionally does not build the sticky plug registry or monthly backtest engine; those become follow-up plans after this seam is stable.

**Tech Stack:** Python 3.13, dataclasses, pandas, DuckDB, pytest, existing `uv run --group fraud` test environment.

---

## Scope Check

The end-state spec covers several independent subsystems: method registry,
selection, plug lifecycle, backtesting, monitoring, reporting, and operator UX.
This plan implements only the method-registry and reusable-selection foundation.

Out of scope for this plan:

- sticky plug lifecycle registry,
- production-facing plug export,
- monthly backtest runner,
- CLI command surface,
- broad file tree reorganization into `methods/`, `findings/`, `validation/`,
  `plugs/`, and `monitoring/`.

Those should follow after the registry and selection seam is proved.

## File Structure

Create:

- `projects/fraud_anomaly_detection/codex_poc/control/discovery/metadata.py`
  - Owns discovery-method semantic metadata and allowed enum values.
- `projects/fraud_anomaly_detection/codex_poc/control/discovery/selection.py`
  - Owns reusable candidate selection and marginal outcome accounting.
- `projects/fraud_anomaly_detection/codex_poc/tests/control/test_discovery_selection.py`
  - Unit tests for marginal selection, dedupe, and metadata-tier exclusion.

Modify:

- `projects/fraud_anomaly_detection/codex_poc/control/discovery/__init__.py`
  - Require discovery methods to expose `metadata`.
- `projects/fraud_anomaly_detection/codex_poc/control/discovery/scenario_method.py`
  - Add metadata to `ScenarioMethod`.
- `projects/fraud_anomaly_detection/codex_poc/control/discovery/graph_method.py`
  - Add metadata to `ResidualRingMethod`.
- `projects/fraud_anomaly_detection/codex_poc/control/discovery/catalog.py`
  - Keep the extension point but make it metadata-aware.
- `projects/fraud_anomaly_detection/codex_poc/control/selected_discovery_report.py`
  - Replace local `_select_graph_methods` logic with `selection.select_candidates`.
- `projects/fraud_anomaly_detection/codex_poc/tests/control/test_discovery.py`
  - Assert metadata is present and catalog filtering remains intuitive.
- `projects/fraud_anomaly_detection/codex_poc/tests/control/test_selected_discovery_report.py`
  - Assert selected report still produces the same high-level shape.
- `projects/fraud_anomaly_detection/codex_poc/README.md`
  - Document method metadata and the enable/add/remove workflow.

## Task 1: Add Discovery Method Metadata

**Files:**
- Create: `projects/fraud_anomaly_detection/codex_poc/control/discovery/metadata.py`
- Modify: `projects/fraud_anomaly_detection/codex_poc/control/discovery/__init__.py`
- Modify: `projects/fraud_anomaly_detection/codex_poc/control/discovery/scenario_method.py`
- Modify: `projects/fraud_anomaly_detection/codex_poc/control/discovery/graph_method.py`
- Test: `projects/fraud_anomaly_detection/codex_poc/tests/control/test_discovery.py`

- [ ] **Step 1: Write the failing metadata tests**

Add these assertions to `test_default_method_catalog_is_the_extension_point` in
`projects/fraud_anomaly_detection/codex_poc/tests/control/test_discovery.py`:

```python
    metadata = [method.metadata for method in methods]
    assert [meta.name for meta in metadata] == [
        "scenario:ring_account_reuse",
        "graph:residual_ring_members",
    ]
    assert metadata[0].method_type == "scenario"
    assert metadata[0].time_semantics == "production_safe"
    assert metadata[0].promotion_tier == "plug_candidate"
    assert metadata[0].enforcement_projection == "scenario_rule"
    assert metadata[1].method_type == "graph"
    assert metadata[1].time_semantics == "snapshot_review"
    assert metadata[1].promotion_tier == "review_queue"
    assert metadata[1].enforcement_projection == "entity_key"
```

Add this assertion to `test_scenario_method_is_a_discovery_method`:

```python
    assert method.metadata.params == {"scenario_name": "ring_account_reuse"}
```

- [ ] **Step 2: Run the tests and verify failure**

Run:

```bash
uv run --group fraud pytest -q \
  projects/fraud_anomaly_detection/codex_poc/tests/control/test_discovery.py
```

Expected: FAIL with an error like
`AttributeError: 'ScenarioMethod' object has no attribute 'metadata'`.

- [ ] **Step 3: Create metadata model**

Create `projects/fraud_anomaly_detection/codex_poc/control/discovery/metadata.py`:

```python
"""Semantic metadata for discovery methods in the fraud-control loop."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

MethodType = Literal["scenario", "graph", "model", "subgroup"]
TimeSemantics = Literal["snapshot_review", "leakfree_asof", "production_safe"]
PromotionTier = Literal["evidence_only", "review_queue", "plug_candidate"]
EnforcementProjection = Literal["entity_key", "scenario_rule", "none"]


@dataclass(frozen=True)
class MethodMetadata:
    """Reproducible semantics for a discovery method."""

    name: str
    version: str
    method_type: MethodType
    time_semantics: TimeSemantics
    promotion_tier: PromotionTier
    enforcement_projection: EnforcementProjection
    enabled: bool = True
    params: dict[str, object] = field(default_factory=dict)
```

- [ ] **Step 4: Update the discovery protocol**

Modify `projects/fraud_anomaly_detection/codex_poc/control/discovery/__init__.py`:

```python
"""Discovery method contract for the fraud-control skeleton."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from projects.fraud_anomaly_detection.codex_poc.control.contract import FindingSet
from projects.fraud_anomaly_detection.codex_poc.control.discovery.metadata import (
    MethodMetadata,
)


@runtime_checkable
class DiscoveryMethod(Protocol):
    name: str
    metadata: MethodMetadata

    def run(self, store: Path | str) -> FindingSet: ...
```

- [ ] **Step 5: Add metadata to `ScenarioMethod`**

Modify `projects/fraud_anomaly_detection/codex_poc/control/discovery/scenario_method.py`
so the class initialization is:

```python
from projects.fraud_anomaly_detection.codex_poc.control.discovery.metadata import (
    MethodMetadata,
)
```

```python
class ScenarioMethod:
    def __init__(self, scenario_name: str):
        self.scenario_name = scenario_name
        self.metadata = MethodMetadata(
            name=f"scenario:{scenario_name}",
            version=SCENARIOS_VERSION,
            method_type="scenario",
            time_semantics="production_safe",
            promotion_tier="plug_candidate",
            enforcement_projection="scenario_rule",
            params={"scenario_name": scenario_name},
        )
        self.name = self.metadata.name
```

Keep the existing `run()` method unchanged except for using `self.name` as it
already does.

- [ ] **Step 6: Add metadata to `ResidualRingMethod`**

Modify `projects/fraud_anomaly_detection/codex_poc/control/discovery/graph_method.py`:

```python
from projects.fraud_anomaly_detection.codex_poc.control.discovery.metadata import (
    MethodMetadata,
)
```

```python
class ResidualRingMethod:
    def __init__(self):
        self.metadata = MethodMetadata(
            name="graph:residual_ring_members",
            version=METHOD_VERSION,
            method_type="graph",
            time_semantics="snapshot_review",
            promotion_tier="review_queue",
            enforcement_projection="entity_key",
            params={"queue": "residual_ring_members"},
        )
        self.name = self.metadata.name
```

Leave `run()` as-is, except it now uses the instance `self.name`.

- [ ] **Step 7: Run the discovery tests**

Run:

```bash
uv run --group fraud pytest -q \
  projects/fraud_anomaly_detection/codex_poc/tests/control/test_discovery.py
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add \
  projects/fraud_anomaly_detection/codex_poc/control/discovery/metadata.py \
  projects/fraud_anomaly_detection/codex_poc/control/discovery/__init__.py \
  projects/fraud_anomaly_detection/codex_poc/control/discovery/scenario_method.py \
  projects/fraud_anomaly_detection/codex_poc/control/discovery/graph_method.py \
  projects/fraud_anomaly_detection/codex_poc/tests/control/test_discovery.py
git commit -m "fraud control: add discovery method metadata"
```

## Task 2: Make The Catalog Metadata-Aware

**Files:**
- Modify: `projects/fraud_anomaly_detection/codex_poc/control/discovery/catalog.py`
- Test: `projects/fraud_anomaly_detection/codex_poc/tests/control/test_discovery.py`

- [ ] **Step 1: Write the failing catalog-profile test**

Add this test to `projects/fraud_anomaly_detection/codex_poc/tests/control/test_discovery.py`:

```python
def test_method_catalog_can_filter_enabled_methods():
    methods = default_methods(enabled_only=True)

    assert all(method.metadata.enabled for method in methods)
    assert [method.metadata.name for method in methods] == [
        "scenario:ring_account_reuse",
        "graph:residual_ring_members",
    ]
```

- [ ] **Step 2: Run the test and verify failure**

Run:

```bash
uv run --group fraud pytest -q \
  projects/fraud_anomaly_detection/codex_poc/tests/control/test_discovery.py::test_method_catalog_can_filter_enabled_methods
```

Expected: FAIL with `TypeError: default_methods() got an unexpected keyword
argument 'enabled_only'`.

- [ ] **Step 3: Update the catalog**

Replace `projects/fraud_anomaly_detection/codex_poc/control/discovery/catalog.py`
with:

```python
"""Discovery method catalog for the walking skeleton.

The catalog is the reviewed extension point for methods that are live in the
control loop. Scenario definitions still live in ``scenarios/register.yaml``;
graph methods live as adapters under ``control.discovery``.
"""
from __future__ import annotations

from projects.fraud_anomaly_detection.codex_poc.control.discovery import DiscoveryMethod
from projects.fraud_anomaly_detection.codex_poc.control.discovery.graph_method import (
    ResidualRingMethod,
)
from projects.fraud_anomaly_detection.codex_poc.control.discovery.scenario_method import (
    ScenarioMethod,
)


def all_methods() -> list[DiscoveryMethod]:
    """Return every reviewed discovery method known to the skeleton."""
    return [ScenarioMethod("ring_account_reuse"), ResidualRingMethod()]


def default_methods(*, enabled_only: bool = True) -> list[DiscoveryMethod]:
    """Return discovery methods enabled for the default skeleton profile."""
    methods = all_methods()
    if not enabled_only:
        return methods
    return [method for method in methods if method.metadata.enabled]
```

- [ ] **Step 4: Run catalog tests**

Run:

```bash
uv run --group fraud pytest -q \
  projects/fraud_anomaly_detection/codex_poc/tests/control/test_discovery.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add \
  projects/fraud_anomaly_detection/codex_poc/control/discovery/catalog.py \
  projects/fraud_anomaly_detection/codex_poc/tests/control/test_discovery.py
git commit -m "fraud control: make discovery catalog metadata-aware"
```

## Task 3: Add Reusable Discovery Selection

**Files:**
- Create: `projects/fraud_anomaly_detection/codex_poc/control/discovery/selection.py`
- Create: `projects/fraud_anomaly_detection/codex_poc/tests/control/test_discovery_selection.py`

- [ ] **Step 1: Write failing selection tests**

Create `projects/fraud_anomaly_detection/codex_poc/tests/control/test_discovery_selection.py`:

```python
from projects.fraud_anomaly_detection.codex_poc.control.discovery.metadata import (
    MethodMetadata,
)
from projects.fraud_anomaly_detection.codex_poc.control.discovery.selection import (
    DiscoveryCandidate,
    SelectionRule,
    select_candidates,
)


def _outcome_factory(bad_users: set[str]):
    def outcome(users: set[str]) -> dict:
        user_ids = {str(user_id) for user_id in users}
        dpd45_users = user_ids & bad_users
        return {
            "users": len(user_ids),
            "dpd45_users": len(dpd45_users),
            "dpd45_user_rate": len(dpd45_users) / len(user_ids) if user_ids else 0.0,
        }

    return outcome


def test_select_candidates_uses_marginal_net_new_after_baseline_and_dedupe():
    baseline = {"u1", "u2"}
    outcome = _outcome_factory({"u3", "u4", "u5"})
    metadata = MethodMetadata(
        name="graph:good",
        version="v1",
        method_type="graph",
        time_semantics="leakfree_asof",
        promotion_tier="plug_candidate",
        enforcement_projection="entity_key",
    )
    duplicate = MethodMetadata(
        name="graph:duplicate",
        version="v1",
        method_type="graph",
        time_semantics="leakfree_asof",
        promotion_tier="plug_candidate",
        enforcement_projection="entity_key",
    )

    result = select_candidates(
        [
            DiscoveryCandidate("graph:good", {"u2", "u3", "u4"}, metadata),
            DiscoveryCandidate("graph:duplicate", {"u3", "u4", "u5"}, duplicate),
        ],
        baseline_users=baseline,
        outcome_fn=outcome,
        rule=SelectionRule(min_marginal_users=2, min_marginal_dpd45_user_rate=0.5),
    )

    assert [row.name for row in result.selected] == ["graph:good"]
    assert [row.name for row in result.excluded] == ["graph:duplicate"]
    assert result.selected[0].marginal_users == {"u3", "u4"}
    assert result.excluded[0].marginal_users == {"u5"}
    assert result.final_users == {"u1", "u2", "u3", "u4"}


def test_select_candidates_excludes_non_promotable_tiers():
    outcome = _outcome_factory({"u10", "u11"})
    metadata = MethodMetadata(
        name="graph:review",
        version="v1",
        method_type="graph",
        time_semantics="snapshot_review",
        promotion_tier="review_queue",
        enforcement_projection="entity_key",
    )

    result = select_candidates(
        [DiscoveryCandidate("graph:review", {"u10", "u11"}, metadata)],
        baseline_users=set(),
        outcome_fn=outcome,
        rule=SelectionRule(min_marginal_users=1, min_marginal_dpd45_user_rate=0.5),
    )

    assert result.selected == []
    assert result.excluded[0].name == "graph:review"
    assert result.excluded[0].reason == "promotion_tier"
    assert result.final_users == set()
```

- [ ] **Step 2: Run the tests and verify failure**

Run:

```bash
uv run --group fraud pytest -q \
  projects/fraud_anomaly_detection/codex_poc/tests/control/test_discovery_selection.py
```

Expected: FAIL with `ModuleNotFoundError` for
`control.discovery.selection`.

- [ ] **Step 3: Implement selection module**

Create `projects/fraud_anomaly_detection/codex_poc/control/discovery/selection.py`:

```python
"""Reusable discovery-candidate selection for promotion into plug derivation."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from projects.fraud_anomaly_detection.codex_poc.control.discovery.metadata import (
    MethodMetadata,
)


OutcomeFn = Callable[[set[str]], dict]


@dataclass(frozen=True)
class DiscoveryCandidate:
    name: str
    users: set[str]
    metadata: MethodMetadata


@dataclass(frozen=True)
class SelectionRule:
    min_marginal_users: int = 10
    min_marginal_dpd45_user_rate: float = 0.50
    promotable_tiers: tuple[str, ...] = ("plug_candidate",)


@dataclass(frozen=True)
class SelectionRow:
    name: str
    users: set[str]
    total: dict
    net_new_users: set[str]
    net: dict
    marginal_users: set[str]
    marginal: dict
    selected: bool
    reason: str
    metadata: MethodMetadata


@dataclass(frozen=True)
class SelectionResult:
    selected: list[SelectionRow]
    excluded: list[SelectionRow]
    final_users: set[str]


def select_candidates(
    candidates: Sequence[DiscoveryCandidate],
    *,
    baseline_users: set[str],
    outcome_fn: OutcomeFn,
    rule: SelectionRule,
) -> SelectionResult:
    """Select candidates by marginal contribution after baseline and prior selections."""
    baseline = {str(user_id) for user_id in baseline_users}
    enriched = []
    for candidate in candidates:
        users = {str(user_id) for user_id in candidate.users}
        net_new_users = users - baseline
        enriched.append(
            {
                "candidate": candidate,
                "users": users,
                "total": outcome_fn(users),
                "net_new_users": net_new_users,
                "net": outcome_fn(net_new_users),
            }
        )

    enriched.sort(
        key=lambda item: (
            item["net"].get("dpd45_user_rate", 0.0),
            item["net"].get("users", 0),
            item["candidate"].name,
        ),
        reverse=True,
    )

    selected: list[SelectionRow] = []
    excluded: list[SelectionRow] = []
    covered = set(baseline)
    for item in enriched:
        candidate = item["candidate"]
        users = item["users"]
        marginal_users = users - covered
        marginal = outcome_fn(marginal_users)
        reason = _exclusion_reason(candidate, marginal, rule)
        include = reason == "selected"
        row = SelectionRow(
            name=candidate.name,
            users=users,
            total=item["total"],
            net_new_users=item["net_new_users"],
            net=item["net"],
            marginal_users=marginal_users,
            marginal=marginal,
            selected=include,
            reason=reason,
            metadata=candidate.metadata,
        )
        if include:
            selected.append(row)
            covered |= users
        else:
            excluded.append(row)
    return SelectionResult(selected=selected, excluded=excluded, final_users=covered)


def _exclusion_reason(
    candidate: DiscoveryCandidate,
    marginal: dict,
    rule: SelectionRule,
) -> str:
    if candidate.metadata.promotion_tier not in rule.promotable_tiers:
        return "promotion_tier"
    if marginal.get("users", 0) < rule.min_marginal_users:
        return "min_marginal_users"
    if marginal.get("dpd45_user_rate", 0.0) < rule.min_marginal_dpd45_user_rate:
        return "min_marginal_dpd45_user_rate"
    return "selected"
```

- [ ] **Step 4: Run selection tests**

Run:

```bash
uv run --group fraud pytest -q \
  projects/fraud_anomaly_detection/codex_poc/tests/control/test_discovery_selection.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add \
  projects/fraud_anomaly_detection/codex_poc/control/discovery/selection.py \
  projects/fraud_anomaly_detection/codex_poc/tests/control/test_discovery_selection.py
git commit -m "fraud control: add reusable discovery selection"
```

## Task 4: Use Selection Module In Selected Discovery Report

**Files:**
- Modify: `projects/fraud_anomaly_detection/codex_poc/control/selected_discovery_report.py`
- Test: `projects/fraud_anomaly_detection/codex_poc/tests/control/test_selected_discovery_report.py`

- [ ] **Step 1: Add selected-report regression assertion**

Modify `projects/fraud_anomaly_detection/codex_poc/tests/control/test_selected_discovery_report.py`
to assert excluded rows now carry reasons:

```python
    assert "reason" in report["excluded_graph_rows"][0]
```

- [ ] **Step 2: Run the selected report test and verify failure**

Run:

```bash
uv run --group fraud pytest -q \
  projects/fraud_anomaly_detection/codex_poc/tests/control/test_selected_discovery_report.py
```

Expected: FAIL with an assertion error because excluded rows do not yet include
`reason`.

- [ ] **Step 3: Import selection primitives**

Add imports to `projects/fraud_anomaly_detection/codex_poc/control/selected_discovery_report.py`:

```python
from projects.fraud_anomaly_detection.codex_poc.control.discovery.metadata import (
    MethodMetadata,
)
from projects.fraud_anomaly_detection.codex_poc.control.discovery.selection import (
    DiscoveryCandidate,
    SelectionRule,
    SelectionRow,
    select_candidates,
)
```

- [ ] **Step 4: Convert graph method dictionaries to candidates**

In `_graph_method_sets`, replace each dictionary shape:

```python
{
    "name": "residual_ring_members",
    "users": set(residual_ring_members(g_scen, flag="scenario_any").user_id.astype(str)),
}
```

with:

```python
DiscoveryCandidate(
    name="residual_ring_members",
    users=set(residual_ring_members(g_scen, flag="scenario_any").user_id.astype(str)),
    metadata=MethodMetadata(
        name="graph:residual_ring_members",
        version="selected-report-1",
        method_type="graph",
        time_semantics="snapshot_review",
        promotion_tier="review_queue",
        enforcement_projection="entity_key",
        params={"source": "selected_discovery_report"},
    ),
)
```

For `high_risk_entity_members_scenario_fraud_seed`, use:

```python
metadata=MethodMetadata(
    name="graph:high_risk_entity_members_scenario_fraud_seed",
    version="selected-report-1",
    method_type="graph",
    time_semantics="snapshot_review",
    promotion_tier="plug_candidate",
    enforcement_projection="entity_key",
    params={"source": "selected_discovery_report"},
)
```

For all other graph screens in this report, use `promotion_tier="review_queue"`
unless the current sample evidence specifically shows they should feed plugs.
This preserves the current selected method and prevents weaker screens from
becoming plug inputs by accident.

- [ ] **Step 5: Replace `_select_graph_methods` implementation**

Delete the body of `_select_graph_methods` and replace it with this adapter:

```python
def _select_graph_methods(
    methods: list[DiscoveryCandidate],
    scenario_union: set[str],
    truth: pd.DataFrame,
    min_marginal_users: int,
    min_marginal_dpd45_user_rate: float,
) -> tuple[list[SelectionRow], list[SelectionRow]]:
    result = select_candidates(
        methods,
        baseline_users=scenario_union,
        outcome_fn=lambda users: _outcome(users, truth),
        rule=SelectionRule(
            min_marginal_users=min_marginal_users,
            min_marginal_dpd45_user_rate=min_marginal_dpd45_user_rate,
        ),
    )
    return result.selected, result.excluded
```

Update selected union construction in `generate_selected_discovery_report`:

```python
selected_graph_union = (
    set().union(*(method.users for method in selected_graphs)) if selected_graphs else set()
)
```

- [ ] **Step 6: Update `_graph_row` for `SelectionRow`**

Replace `_graph_row` with:

```python
def _graph_row(method: SelectionRow) -> dict:
    return {
        "graph method": method.name,
        "total users / DPD45": (
            f"{method.total['users']:,} / {_pct(method.total['dpd45_user_rate'])}"
        ),
        "net-new beyond scenarios / DPD45": (
            f"{method.net['users']:,} / {_pct(method.net['dpd45_user_rate'])}"
        ),
        "marginal after dedupe / DPD45": (
            f"{method.marginal['users']:,} / {_pct(method.marginal['dpd45_user_rate'])}"
        ),
        "selected?": "yes" if method.selected else "no",
        "reason": method.reason,
    }
```

Update both markdown table calls for graph rows to include `"reason"` in the
headers.

- [ ] **Step 7: Run selected-report test**

Run:

```bash
uv run --group fraud pytest -q \
  projects/fraud_anomaly_detection/codex_poc/tests/control/test_selected_discovery_report.py
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add \
  projects/fraud_anomaly_detection/codex_poc/control/selected_discovery_report.py \
  projects/fraud_anomaly_detection/codex_poc/tests/control/test_selected_discovery_report.py
git commit -m "fraud control: reuse discovery selection in selected report"
```

## Task 5: Enforce Metadata-Aware Runner Inputs

**Files:**
- Modify: `projects/fraud_anomaly_detection/codex_poc/tests/control/test_run.py`
- Modify: `projects/fraud_anomaly_detection/codex_poc/control/run.py`

- [ ] **Step 1: Update static test method metadata**

In `projects/fraud_anomaly_detection/codex_poc/tests/control/test_run.py`, add:

```python
from projects.fraud_anomaly_detection.codex_poc.control.discovery.metadata import (
    MethodMetadata,
)
```

Then update `StaticMethod`:

```python
class StaticMethod:
    name = "test:static"
    metadata = MethodMetadata(
        name="test:static",
        version="v1",
        method_type="model",
        time_semantics="leakfree_asof",
        promotion_tier="plug_candidate",
        enforcement_projection="entity_key",
    )

    def run(self, store):
        return FindingSet(
            method=self.name,
            method_version=self.metadata.version,
            findings=[Finding("u1"), Finding("u2"), Finding("u3")],
        )
```

- [ ] **Step 2: Add a rejection test for legacy methods**

Add this test to
`projects/fraud_anomaly_detection/codex_poc/tests/control/test_run.py`:

```python
class LegacyMethodWithoutMetadata:
    name = "test:legacy"

    def run(self, store):
        return FindingSet(
            method=self.name,
            method_version="legacy",
            findings=[Finding("u1")],
        )


def test_run_skeleton_rejects_methods_without_metadata(tiny_store, tmp_path):
    with pytest.raises(TypeError, match="metadata"):
        run_skeleton(
            tiny_store,
            findings_db=tmp_path / "findings.duckdb",
            methods=[LegacyMethodWithoutMetadata()],
        )
```

- [ ] **Step 3: Run runner tests and verify failure**

Run:

```bash
uv run --group fraud pytest -q \
  projects/fraud_anomaly_detection/codex_poc/tests/control/test_run.py
```

Expected: FAIL because `run_skeleton` does not yet reject legacy methods
without metadata.

- [ ] **Step 4: Validate methods at runner entry**

Add this helper to `projects/fraud_anomaly_detection/codex_poc/control/run.py`:

```python
def _validate_methods(methods: Sequence[DiscoveryMethod]) -> None:
    for method in methods:
        metadata = getattr(method, "metadata", None)
        if metadata is None:
            raise TypeError(
                f"Discovery method {getattr(method, 'name', method)!r} is missing metadata"
            )
        if metadata.name != method.name:
            raise ValueError(
                f"Discovery method name {method.name!r} does not match metadata "
                f"name {metadata.name!r}"
            )
```

Call it immediately after `active_methods` is computed in `run_skeleton`:

```python
    active_methods = list(methods) if methods is not None else default_methods()
    _validate_methods(active_methods)
```

- [ ] **Step 5: Run runner tests**

Run:

```bash
uv run --group fraud pytest -q \
  projects/fraud_anomaly_detection/codex_poc/tests/control/test_run.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add projects/fraud_anomaly_detection/codex_poc/control/run.py \
  projects/fraud_anomaly_detection/codex_poc/tests/control/test_run.py
git commit -m "fraud control: require discovery method metadata"
```

## Task 6: Update Operator Documentation

**Files:**
- Modify: `projects/fraud_anomaly_detection/codex_poc/README.md`
- Modify: `projects/fraud_anomaly_detection/codex_poc/tests/control/test_operator_docs.py`

- [ ] **Step 1: Write failing doc test assertions**

Add these assertions to
`projects/fraud_anomaly_detection/codex_poc/tests/control/test_operator_docs.py`:

```python
    assert "method metadata" in text
    assert "promotion_tier" in text
    assert "Disable a method" in text
    assert "selected-discovery report uses reusable selection" in text
```

- [ ] **Step 2: Run doc test and verify failure**

Run:

```bash
uv run --group fraud pytest -q \
  projects/fraud_anomaly_detection/codex_poc/tests/control/test_operator_docs.py
```

Expected: FAIL because the README does not yet document the new metadata and
selection seam.

- [ ] **Step 3: Update README method sections**

In `projects/fraud_anomaly_detection/codex_poc/README.md`, after the paragraph
that says `The default method list lives in control/discovery/catalog.py`, add:

```markdown
Each discovery method now exposes method metadata:

- `method_type`: scenario, graph, model, or subgroup.
- `time_semantics`: snapshot_review, leakfree_asof, or production_safe.
- `promotion_tier`: evidence_only, review_queue, or plug_candidate.
- `enforcement_projection`: entity_key, scenario_rule, or none.

This keeps broad discovery safe: a method can be useful for review without
being eligible for plug derivation. Disable a method by removing it from
`default_methods()` or by setting `enabled=False` in its metadata.

The selected-discovery report uses reusable selection logic. It screens graph
methods by marginal net-new contribution after the scenario baseline and records
why each graph method was selected or excluded.
```

- [ ] **Step 4: Run doc test**

Run:

```bash
uv run --group fraud pytest -q \
  projects/fraud_anomaly_detection/codex_poc/tests/control/test_operator_docs.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add \
  projects/fraud_anomaly_detection/codex_poc/README.md \
  projects/fraud_anomaly_detection/codex_poc/tests/control/test_operator_docs.py
git commit -m "fraud control: document method metadata workflow"
```

## Task 7: Run Focused And Full Verification

**Files:**
- No code changes expected.

- [ ] **Step 1: Run focused control tests**

Run:

```bash
uv run --group fraud pytest -q \
  projects/fraud_anomaly_detection/codex_poc/tests/control \
  projects/fraud_anomaly_detection/tests/test_graph_discover.py \
  projects/fraud_anomaly_detection/tests/test_subgroup_core.py
```

Expected: PASS.

- [ ] **Step 2: Run focused ruff**

Run:

```bash
uv run --group fraud ruff check \
  projects/fraud_anomaly_detection/codex_poc/control \
  projects/fraud_anomaly_detection/codex_poc/tests/control
```

Expected: PASS.

- [ ] **Step 3: Run archived-import guard**

Run:

```bash
python - <<'PY'
import subprocess
import sys

result = subprocess.run(
    [
        "rg",
        "codex_poc\\\\.archived|from .*archived|import .*archived",
        "projects/fraud_anomaly_detection/codex_poc/control",
    ],
    capture_output=True,
    text=True,
)
if result.returncode == 0:
    print(result.stdout)
    sys.exit(1)
if result.returncode == 1:
    print("No archived imports found")
    sys.exit(0)
print(result.stderr)
sys.exit(result.returncode)
PY
```

Expected: `No archived imports found`.

- [ ] **Step 4: Run full fraud test suite if focused tests pass**

Run:

```bash
uv run --group fraud pytest -q
```

Expected: PASS. If this is too slow for the implementation session, record the
reason and at minimum include the focused test results above.

- [ ] **Step 5: Final commit if verification changed docs or snapshots**

If no files changed during verification, do not commit.

If formatting or docs changed:

```bash
git status --short
git add <changed-files>
git commit -m "fraud control: finalize registry selection foundation"
```

## Follow-Up Plans After This One

1. Sticky plug registry lifecycle: proposed, active, expired, rejected.
2. First-class monthly backtesting module using the same selection and plug
   validation surfaces.
3. Operator command surface for methods, runs, reports, plugs, export, and
   backtests.
4. Directory reorganization toward the end-state structure once the seams are
   stable and tests cover behavior.
