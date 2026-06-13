# Fraud Control System — Walking Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an end-to-end *walking skeleton* of the fraud-control system — discovery → finding store → plug derivation → holdout → monitoring — with one real scenario method and one real graph method wired through a stable contract, running on the local sample store.

**Architecture:** A new `codex_poc/control/` subpackage. Discovery methods (adapters over the existing `scenarios` register and `graph` discovery functions) emit findings in one contract shape into a DuckDB-backed finding store (dated, version-tagged snapshots). Plug derivation reads findings + the graph store to produce a validated burned-key list via extract → validate → qualify, with all thresholds in one config. A two-state holdout harness and a monitoring module measure the loop. See `codex_poc/CONTROL_SYSTEM_DESIGN.md` for the design and `PRINCIPLES.md` for the principles this answers to.

**Tech Stack:** Python, DuckDB (store + finding snapshots), pandas, igraph (via the existing `graph/` library), pytest. Run everything with `uv run --group fraud ...`.

**Scope note:** This is deliberately the *minimal* end-to-end slice (one scenario method, one graph method) to prove the contract and the loop. Additional discovery methods, threshold tuning, and the warehouse-facing plug table are explicitly out of scope — they plug into this skeleton later (most need v3 data / VPN). Thresholds here are placeholder defaults, not tuned.

**Sample store:** `projects/fraud_anomaly_detection/data/graph/fraud_graph.duckdb` — tables `advances` (user_id, is_fraud, label_gross_dpd45, label_mature_d45, feature_as_of_ts, loan_amount, scenario trigger cols), `users` (user_id, n_advances, n_mature_advances, n_bad_advances, bad_advance_rate, first_seen_ts, last_seen_ts, identity_created_time), `entities` (entity_type, entity_value, n_users, n_advances, first/last_seen_ts), `edges` (user_id, entity_type, entity_value, ts, source).

---

## File Structure

```
codex_poc/control/
├── __init__.py
├── contract.py          # Finding, FindingSet — the discovery output contract
├── config.py            # ControlConfig — all tunable thresholds in one place
├── finding_store.py     # DuckDB snapshot store: write/read, version tags, trim-when-unchanged
├── discovery/
│   ├── __init__.py      # DiscoveryMethod protocol + REGISTRY
│   ├── scenario_method.py   # adapter over scenarios.assign
│   └── graph_method.py      # adapter over graph.discover.residual_ring_members
├── plug.py              # extract → validate → qualify; BurnedKey
├── holdout.py           # two-state (A / A+month) derive+evaluate harness
├── monitoring.py        # discovery rate / prevention / leakage stats
└── run.py               # orchestrator: discovery → store → plugs

codex_poc/tests/control/
├── conftest.py          # tiny synthetic DuckDB store fixture
├── test_contract.py
├── test_finding_store.py
├── test_discovery.py
├── test_plug.py
├── test_holdout.py
├── test_monitoring.py
└── test_run.py
```

Each file has one responsibility; findings flow through `contract.Finding`, the single seam.

---

## Task 1: Discovery contract + config

**Files:**
- Create: `codex_poc/control/__init__.py` (empty)
- Create: `codex_poc/control/contract.py`
- Create: `codex_poc/control/config.py`
- Test: `codex_poc/tests/control/test_contract.py`

- [ ] **Step 1: Write the failing test**

```python
# codex_poc/tests/control/test_contract.py
from projects.fraud_anomaly_detection.codex_poc.control.contract import Finding, FindingSet
from projects.fraud_anomaly_detection.codex_poc.control.config import ControlConfig


def test_finding_set_to_frame_has_contract_columns():
    fs = FindingSet(
        method="scenario:ring_device_burst",
        method_version="2026-06-08.2",
        findings=[Finding(user_id="u1", score=1.0, evidence={"scenario": "ring_device_burst"})],
    )
    df = fs.to_frame()
    assert list(df.columns) == ["user_id", "method", "method_version", "score", "evidence"]
    assert df.iloc[0]["user_id"] == "u1"
    assert df.iloc[0]["method"] == "scenario:ring_device_burst"


def test_config_defaults_are_tunable_in_one_place():
    cfg = ControlConfig()
    assert cfg.block_tier_precision == 0.8
    assert cfg.min_support >= 1
    assert cfg.min_coverage >= 1
    assert cfg.min_corroborating_types >= 1
    assert cfg.holdout_days == 30
    cfg2 = ControlConfig(block_tier_precision=0.7)
    assert cfg2.block_tier_precision == 0.7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --group fraud pytest projects/fraud_anomaly_detection/codex_poc/tests/control/test_contract.py -v`
Expected: FAIL (ModuleNotFoundError: control.contract).

- [ ] **Step 3: Write minimal implementation**

```python
# codex_poc/control/contract.py
"""The discovery output contract — every method emits findings in this shape."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

CONTRACT_COLUMNS = ["user_id", "method", "method_version", "score", "evidence"]


@dataclass(frozen=True)
class Finding:
    """One discovered-suspect user, with the evidence that surfaced it."""
    user_id: str
    score: float = 1.0          # method-local rank/strength; not comparable across methods
    evidence: dict = field(default_factory=dict)


@dataclass(frozen=True)
class FindingSet:
    """All findings from one method run, tagged with the method's identity+version."""
    method: str                 # e.g. "scenario:ring_device_burst" or "graph:residual_ring_members"
    method_version: str
    findings: list[Finding]

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {
                "user_id": str(f.user_id),
                "method": self.method,
                "method_version": self.method_version,
                "score": float(f.score),
                "evidence": f.evidence,
            }
            for f in self.findings
        ]
        return pd.DataFrame(rows, columns=CONTRACT_COLUMNS)
```

```python
# codex_poc/control/config.py
"""All tunable thresholds in one place (PRINCIPLES P6) — never baked into facts."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ControlConfig:
    block_tier_precision: float = 0.8     # tau: min DPD45 precision for a plug
    min_support: int = 3                  # min users touching a key to consider it
    min_coverage: int = 2                 # min discovered-fraud users a key must cover
    min_corroborating_types: int = 2      # multi-key corroboration bar
    holdout_days: int = 30                # the two-state holdout window
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --group fraud pytest projects/fraud_anomaly_detection/codex_poc/tests/control/test_contract.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add projects/fraud_anomaly_detection/codex_poc/control/ projects/fraud_anomaly_detection/codex_poc/tests/control/test_contract.py
git commit -m "fraud control: discovery contract + tunable config scaffold"
```

---

## Task 2: Finding store (versioned snapshots, trim-when-unchanged)

**Files:**
- Create: `codex_poc/control/finding_store.py`
- Create: `codex_poc/tests/control/conftest.py`
- Test: `codex_poc/tests/control/test_finding_store.py`

- [ ] **Step 1: Write the conftest fixture**

```python
# codex_poc/tests/control/conftest.py
"""A tiny synthetic DuckDB store with the four control-relevant tables."""
import duckdb
import pandas as pd
import pytest


@pytest.fixture
def tiny_store(tmp_path):
    path = tmp_path / "tiny.duckdb"
    con = duckdb.connect(str(path))
    advances = pd.DataFrame({
        "user_id": ["u1", "u2", "u3", "u4", "u5"],
        "is_fraud": [True, False, False, False, False],
        "label_gross_dpd45": [True, True, True, False, False],
        "label_mature_d45": [True, True, True, True, True],
        "feature_as_of_ts": pd.to_datetime(
            ["2026-01-01", "2026-01-02", "2026-02-15", "2026-02-20", "2026-01-03"]),
    })
    edges = pd.DataFrame({
        "user_id": ["u1", "u2", "u3", "u4", "u5"],
        "entity_type": ["bank", "bank", "bank", "device", "device"],
        "entity_value": ["acctA", "acctA", "acctA", "devX", "devX"],
        "ts": pd.to_datetime(
            ["2026-01-01", "2026-01-02", "2026-02-15", "2026-02-20", "2026-01-03"]),
        "source": ["advance"] * 5,
    })
    con.execute("CREATE TABLE advances AS SELECT * FROM advances")
    con.execute("CREATE TABLE edges AS SELECT * FROM edges")
    con.close()
    return path
```

- [ ] **Step 2: Write the failing test**

```python
# codex_poc/tests/control/test_finding_store.py
import pandas as pd
from projects.fraud_anomaly_detection.codex_poc.control.contract import Finding, FindingSet
from projects.fraud_anomaly_detection.codex_poc.control.finding_store import FindingStore


def _fs(method="scenario:s", users=("u1",)):
    return FindingSet(method, "v1", [Finding(u) for u in users])


def test_write_then_read_latest_roundtrips(tmp_path):
    store = FindingStore(tmp_path / "findings.duckdb")
    store.write_snapshot("2026-06-13", data_version="v3", finding_sets=[_fs(users=["u1", "u2"])])
    latest = store.read_latest()
    assert set(latest["user_id"]) == {"u1", "u2"}
    assert latest.iloc[0]["refresh_key"] == "2026-06-13"
    assert latest.iloc[0]["data_version"] == "v3"


def test_unchanged_snapshot_is_trimmed(tmp_path):
    store = FindingStore(tmp_path / "findings.duckdb")
    store.write_snapshot("2026-06-13", data_version="v3", finding_sets=[_fs(users=["u1"])])
    wrote = store.write_snapshot("2026-06-14", data_version="v3", finding_sets=[_fs(users=["u1"])])
    assert wrote is False                      # identical content → skipped
    assert store.refresh_keys() == ["2026-06-13"]


def test_changed_snapshot_is_kept(tmp_path):
    store = FindingStore(tmp_path / "findings.duckdb")
    store.write_snapshot("2026-06-13", data_version="v3", finding_sets=[_fs(users=["u1"])])
    wrote = store.write_snapshot("2026-06-14", data_version="v3", finding_sets=[_fs(users=["u1", "u2"])])
    assert wrote is True
    assert store.refresh_keys() == ["2026-06-13", "2026-06-14"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run --group fraud pytest projects/fraud_anomaly_detection/codex_poc/tests/control/test_finding_store.py -v`
Expected: FAIL (cannot import FindingStore).

- [ ] **Step 4: Write minimal implementation**

```python
# codex_poc/control/finding_store.py
"""DuckDB-backed finding store: dated, version-tagged snapshots; trim-when-unchanged.

Findings are log-like (PRINCIPLES P5/P6 + CONTROL_SYSTEM_DESIGN §4): every run
appends a snapshot keyed by refresh + logic-version + data-version. An identical
snapshot (same content as the latest) is skipped so the trail stays lean.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd

from projects.fraud_anomaly_detection.codex_poc.control.contract import FindingSet

_SCHEMA = """
CREATE TABLE IF NOT EXISTS findings (
    refresh_key   VARCHAR,
    data_version  VARCHAR,
    content_hash  VARCHAR,
    method        VARCHAR,
    method_version VARCHAR,
    user_id       VARCHAR,
    score         DOUBLE,
    evidence      VARCHAR
)
"""


class FindingStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        with duckdb.connect(str(self.path)) as con:
            con.execute(_SCHEMA)

    def _frame(self, finding_sets: list[FindingSet]) -> pd.DataFrame:
        if not finding_sets:
            return pd.DataFrame(columns=["method", "method_version", "user_id", "score", "evidence"])
        out = pd.concat([fs.to_frame() for fs in finding_sets], ignore_index=True)
        out["evidence"] = out["evidence"].map(lambda e: json.dumps(e, sort_keys=True, default=str))
        return out

    @staticmethod
    def _hash(df: pd.DataFrame) -> str:
        canon = df[["method", "user_id", "score", "evidence"]].sort_values(
            ["method", "user_id"]).to_csv(index=False)
        return hashlib.sha256(canon.encode()).hexdigest()

    def refresh_keys(self) -> list[str]:
        with duckdb.connect(str(self.path), read_only=True) as con:
            rows = con.execute(
                "SELECT DISTINCT refresh_key FROM findings ORDER BY refresh_key").fetchall()
        return [r[0] for r in rows]

    def _latest_hash(self) -> str | None:
        keys = self.refresh_keys()
        if not keys:
            return None
        with duckdb.connect(str(self.path), read_only=True) as con:
            [(h,)] = con.execute(
                "SELECT content_hash FROM findings WHERE refresh_key = ? LIMIT 1", [keys[-1]]
            ).fetchall()
        return h

    def write_snapshot(self, refresh_key: str, data_version: str,
                       finding_sets: list[FindingSet]) -> bool:
        """Append a snapshot. Returns False (and writes nothing) if identical to the latest."""
        df = self._frame(finding_sets)
        content_hash = self._hash(df)
        if content_hash == self._latest_hash():
            return False
        df = df.assign(refresh_key=refresh_key, data_version=data_version, content_hash=content_hash)
        with duckdb.connect(str(self.path)) as con:
            con.execute("INSERT INTO findings SELECT refresh_key, data_version, content_hash, "
                        "method, method_version, user_id, score, evidence FROM df")
        return True

    def read_latest(self) -> pd.DataFrame:
        keys = self.refresh_keys()
        if not keys:
            return pd.DataFrame(columns=["refresh_key", "data_version", "method",
                                         "method_version", "user_id", "score", "evidence"])
        with duckdb.connect(str(self.path), read_only=True) as con:
            return con.execute("SELECT refresh_key, data_version, method, method_version, "
                               "user_id, score, evidence FROM findings WHERE refresh_key = ?",
                               [keys[-1]]).df()
```

- [ ] **Step 5: Run tests, verify pass, commit**

Run: `uv run --group fraud pytest projects/fraud_anomaly_detection/codex_poc/tests/control/test_finding_store.py -v`
Expected: PASS (3 passed).

```bash
git add projects/fraud_anomaly_detection/codex_poc/control/finding_store.py projects/fraud_anomaly_detection/codex_poc/tests/control/
git commit -m "fraud control: finding store with version-tagged, trim-when-unchanged snapshots"
```

---

## Task 3: Discovery method protocol + scenario adapter

**Files:**
- Create: `codex_poc/control/discovery/__init__.py`
- Create: `codex_poc/control/discovery/scenario_method.py`
- Test: `codex_poc/tests/control/test_discovery.py` (scenario part)

- [ ] **Step 1: Write the failing test**

```python
# codex_poc/tests/control/test_discovery.py
from pathlib import Path

import pytest

from projects.fraud_anomaly_detection.codex_poc.control.discovery import DiscoveryMethod
from projects.fraud_anomaly_detection.codex_poc.control.discovery.scenario_method import (
    ScenarioMethod,
)

SAMPLE = Path("projects/fraud_anomaly_detection/data/graph/fraud_graph.duckdb")


def test_scenario_method_is_a_discovery_method():
    method = ScenarioMethod(scenario_name="ring_account_reuse")
    assert isinstance(method, DiscoveryMethod)
    assert method.name == "scenario:ring_account_reuse"


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample store not built")
def test_scenario_method_emits_contract_findings_on_sample():
    # Run against the real sample store: its `advances` carry the scenario
    # trigger columns `assign()` needs (the tiny synthetic store does not).
    fs = ScenarioMethod("ring_account_reuse").run(SAMPLE)
    assert fs.method == "scenario:ring_account_reuse"
    assert fs.method_version                 # non-empty register version
    assert len(fs.findings) > 0              # ring_account_reuse fires on the sample
    for f in fs.findings:
        assert f.evidence["scenario"] == "ring_account_reuse"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --group fraud pytest projects/fraud_anomaly_detection/codex_poc/tests/control/test_discovery.py -v`
Expected: FAIL (cannot import DiscoveryMethod).

- [ ] **Step 3: Write minimal implementation**

```python
# codex_poc/control/discovery/__init__.py
"""The discovery-method contract: any method runs against the store and emits a FindingSet."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from projects.fraud_anomaly_detection.codex_poc.control.contract import FindingSet


@runtime_checkable
class DiscoveryMethod(Protocol):
    name: str

    def run(self, store: Path | str) -> FindingSet: ...
```

```python
# codex_poc/control/discovery/scenario_method.py
"""Adapter: a registered scenario becomes a discovery method (read register, emit findings).

Consumes the canonical scenario definitions (PRINCIPLES P8 / CONTROL_SYSTEM_DESIGN §2)
without wrapping engine.py's API beyond the bound `assign`.
"""
from __future__ import annotations

from pathlib import Path

import duckdb

from projects.fraud_anomaly_detection.codex_poc.control.contract import Finding, FindingSet
from projects.fraud_anomaly_detection.scenarios import SCENARIOS_VERSION, assign


class ScenarioMethod:
    def __init__(self, scenario_name: str):
        self.scenario_name = scenario_name
        self.name = f"scenario:{scenario_name}"

    def run(self, store: Path | str) -> FindingSet:
        with duckdb.connect(str(store), read_only=True) as con:
            advances = con.execute("SELECT * FROM advances").df()
        flagged = assign(advances)
        col = f"scenario_{self.scenario_name}"
        hit_users = (flagged.loc[flagged[col].fillna(False), "user_id"]
                     .astype(str).unique().tolist())
        findings = [Finding(user_id=u, evidence={"scenario": self.scenario_name})
                    for u in hit_users]
        return FindingSet(self.name, SCENARIOS_VERSION, findings)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --group fraud pytest projects/fraud_anomaly_detection/codex_poc/tests/control/test_discovery.py -v`
Expected: the protocol test PASSes; the sample test PASSes (or SKIPs if the sample store isn't built — build it with `uv run --group fraud python -m projects.fraud_anomaly_detection.analysis.graph_store_demo`).

- [ ] **Step 5: Commit**

```bash
git add projects/fraud_anomaly_detection/codex_poc/control/discovery/ projects/fraud_anomaly_detection/codex_poc/tests/control/test_discovery.py
git commit -m "fraud control: discovery-method protocol + scenario adapter"
```

---

## Task 4: Graph discovery adapter

**Files:**
- Create: `codex_poc/control/discovery/graph_method.py`
- Test: extend `codex_poc/tests/control/test_discovery.py`

- [ ] **Step 1: Write the failing test (against the real sample store)**

```python
# add to codex_poc/tests/control/test_discovery.py (Path, pytest, SAMPLE already imported in Task 3)
from projects.fraud_anomaly_detection.codex_poc.control.discovery.graph_method import (
    ResidualRingMethod,
)


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample store not built")
def test_graph_method_emits_contract_findings_on_sample():
    fs = ResidualRingMethod().run(SAMPLE)
    assert fs.method == "graph:residual_ring_members"
    assert len(fs.findings) > 0
    f = fs.findings[0]
    assert isinstance(f.user_id, str)
    assert "ring_users" in f.evidence and "entity_types" in f.evidence
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --group fraud pytest projects/fraud_anomaly_detection/codex_poc/tests/control/test_discovery.py -k graph_method -v`
Expected: FAIL (cannot import ResidualRingMethod).

- [ ] **Step 3: Write minimal implementation**

```python
# codex_poc/control/discovery/graph_method.py
"""Adapter: an existing graph discovery queue becomes a discovery method.

Wraps graph.discover.residual_ring_members (the multi-type ring queue) and
re-shapes its frame into contract Findings. Other graph queues plug in the
same way (suspicion_queue, etc.) — one adapter each.
"""
from __future__ import annotations

from pathlib import Path

from projects.fraud_anomaly_detection.codex_poc.control.contract import Finding, FindingSet
from projects.fraud_anomaly_detection.graph.discover import residual_ring_members
from projects.fraud_anomaly_detection.graph.load import load_graph

METHOD_VERSION = "graph-skeleton-1"


class ResidualRingMethod:
    name = "graph:residual_ring_members"

    def run(self, store: Path | str) -> FindingSet:
        g = load_graph(store)                      # default layers + scenario overlay
        queue = residual_ring_members(g)           # user_id + ring evidence columns
        findings = [
            Finding(
                user_id=str(row.user_id),
                score=float(row.ring_flagged),     # association strength as the local score
                evidence={"comp_id": int(row.comp_id), "ring_users": int(row.ring_users),
                          "ring_types": int(row.ring_types), "entity_types": row.entity_types},
            )
            for row in queue.itertuples(index=False)
        ]
        return FindingSet(self.name, METHOD_VERSION, findings)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --group fraud pytest projects/fraud_anomaly_detection/codex_poc/tests/control/test_discovery.py -k graph_method -v`
Expected: PASS (or SKIP if the sample store isn't built — build it first with `uv run --group fraud python -m projects.fraud_anomaly_detection.analysis.graph_store_demo` if needed).

- [ ] **Step 5: Commit**

```bash
git add projects/fraud_anomaly_detection/codex_poc/control/discovery/graph_method.py projects/fraud_anomaly_detection/codex_poc/tests/control/test_discovery.py
git commit -m "fraud control: graph discovery adapter (residual ring members)"
```

---

## Task 5: Plug derivation — extract → validate → qualify

**Files:**
- Create: `codex_poc/control/plug.py`
- Test: `codex_poc/tests/control/test_plug.py`

The three stages (CONTROL_SYSTEM_DESIGN §5): **extract** candidate keys from findings (mechanical), **validate** each into a stats row (precision vs DPD45, coverage vs discovery, volume, innocents — facts), **qualify** with the config thresholds (cheap filter). Extract+validate are one pass producing a persisted candidate-stats frame; qualify is the cheap re-runnable filter.

- [ ] **Step 1: Write the failing test**

```python
# codex_poc/tests/control/test_plug.py
import duckdb
import pandas as pd
from projects.fraud_anomaly_detection.codex_poc.control.config import ControlConfig
from projects.fraud_anomaly_detection.codex_poc.control import plug


def test_candidate_stats_compute_precision_and_coverage(tiny_store):
    # discovered fraud = u1,u2,u3 (the acctA ring). acctA: 3 users, all DPD45 → precision 1.0.
    discovered = pd.Series(["u1", "u2", "u3"])
    stats = plug.candidate_stats(tiny_store, discovered)
    acctA = stats[(stats.entity_type == "bank") & (stats.entity_value == "acctA")].iloc[0]
    assert acctA.support == 3
    assert acctA.dpd45_precision == 1.0
    assert acctA.coverage == 3          # covers all 3 discovered-fraud users
    assert acctA.innocents == 0


def test_qualify_filters_by_config_over_stats(tiny_store):
    discovered = pd.Series(["u1", "u2", "u3"])
    stats = plug.candidate_stats(tiny_store, discovered)
    keys = plug.qualify(stats, ControlConfig(min_support=3, min_coverage=2,
                                             block_tier_precision=0.8))
    assert ("bank", "acctA") in set(zip(keys.entity_type, keys.entity_value))
    # retune precision up past acctA's value → it drops, with no re-extract
    none = plug.qualify(stats, ControlConfig(block_tier_precision=1.01))
    assert len(none) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --group fraud pytest projects/fraud_anomaly_detection/codex_poc/tests/control/test_plug.py -v`
Expected: FAIL (cannot import plug.candidate_stats).

- [ ] **Step 3: Write minimal implementation**

```python
# codex_poc/control/plug.py
"""Plug derivation: extract -> validate -> qualify (CONTROL_SYSTEM_DESIGN §5).

candidate_stats() is the expensive extract+validate pass and yields FACTS;
qualify() is the cheap, re-runnable threshold filter over those facts (P6).
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


def candidate_stats(store: Path | str, discovered_users: pd.Series) -> pd.DataFrame:
    """One row per shared key (entity), with the dual-validation facts.

    precision = DPD45 rate among mature advances touching the key (over-reach check);
    coverage  = # discovered-fraud users touching the key (under-cover check);
    support   = distinct users on the key; innocents = mature non-DPD45 users on it.
    """
    discovered = set(discovered_users.astype(str))
    with duckdb.connect(str(store), read_only=True) as con:
        edge_users = con.execute(
            "SELECT entity_type, entity_value, CAST(user_id AS VARCHAR) AS user_id "
            "FROM edges GROUP BY 1,2,3").df()
        outcomes = con.execute(
            "SELECT CAST(user_id AS VARCHAR) AS user_id, "
            "max(CASE WHEN label_mature_d45 THEN 1 ELSE 0 END) AS mature, "
            "max(CASE WHEN label_mature_d45 AND label_gross_dpd45 THEN 1 ELSE 0 END) AS bad "
            "FROM advances GROUP BY 1").df()
    df = edge_users.merge(outcomes, on="user_id", how="left").fillna({"mature": 0, "bad": 0})
    df["discovered"] = df["user_id"].isin(discovered)
    g = df.groupby(["entity_type", "entity_value"])
    stats = g.agg(
        support=("user_id", "nunique"),
        mature_users=("mature", "sum"),
        bad_users=("bad", "sum"),
        coverage=("discovered", "sum"),
    ).reset_index()
    stats["dpd45_precision"] = (stats.bad_users / stats.mature_users.where(stats.mature_users > 0))
    stats["dpd45_precision"] = stats["dpd45_precision"].fillna(0.0)
    stats["innocents"] = stats.mature_users - stats.bad_users
    return stats


def qualify(stats: pd.DataFrame, config) -> pd.DataFrame:
    """Cheap conjunctive filter over the stats facts → the burned-key list."""
    keep = (
        (stats.support >= config.min_support)
        & (stats.coverage >= config.min_coverage)
        & (stats.dpd45_precision >= config.block_tier_precision)
    )
    return stats.loc[keep].sort_values(
        ["dpd45_precision", "coverage", "support"], ascending=False, kind="stable"
    ).reset_index(drop=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --group fraud pytest projects/fraud_anomaly_detection/codex_poc/tests/control/test_plug.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add projects/fraud_anomaly_detection/codex_poc/control/plug.py projects/fraud_anomaly_detection/codex_poc/tests/control/test_plug.py
git commit -m "fraud control: plug derivation (extract/validate facts + cheap qualify filter)"
```

---

## Task 6: Two-state holdout harness

**Files:**
- Create: `codex_poc/control/holdout.py`
- Test: `codex_poc/tests/control/test_holdout.py`

State A = advances with `feature_as_of_ts` before the cutoff; State B = all. Cutoff = newest `feature_as_of_ts` minus `holdout_days` (anchored to the store, not wall clock — matching `graph.discover.fresh_rings`). The harness returns the cutoff and the held-out user set so derivation can be restricted to A and evaluation to the delta.

- [ ] **Step 1: Write the failing test**

```python
# codex_poc/tests/control/test_holdout.py
from projects.fraud_anomaly_detection.codex_poc.control.config import ControlConfig
from projects.fraud_anomaly_detection.codex_poc.control import holdout


def test_split_partitions_users_by_cutoff(tiny_store):
    # tiny_store newest ts = 2026-02-20; 30-day holdout cutoff ~2026-01-21.
    split = holdout.two_state_split(tiny_store, ControlConfig(holdout_days=30))
    assert set(split.state_a_users) == {"u1", "u2", "u5"}     # Jan dates
    assert set(split.holdout_users) == {"u3", "u4"}           # mid-Feb dates
    assert split.cutoff < split.newest
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --group fraud pytest projects/fraud_anomaly_detection/codex_poc/tests/control/test_holdout.py -v`
Expected: FAIL (cannot import holdout.two_state_split).

- [ ] **Step 3: Write minimal implementation**

```python
# codex_poc/control/holdout.py
"""Two-state leak-free split (PRINCIPLES P7 / CONTROL_SYSTEM_DESIGN §3).

State A = advances at/before the cutoff (derive here); the held-out delta =
advances after it (evaluate here). Cutoff anchored to the store's newest advance.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd


@dataclass(frozen=True)
class TwoStateSplit:
    cutoff: pd.Timestamp
    newest: pd.Timestamp
    state_a_users: list[str]
    holdout_users: list[str]


def two_state_split(store: Path | str, config) -> TwoStateSplit:
    with duckdb.connect(str(store), read_only=True) as con:
        adv = con.execute(
            "SELECT CAST(user_id AS VARCHAR) AS user_id, feature_as_of_ts FROM advances").df()
    adv["feature_as_of_ts"] = pd.to_datetime(adv["feature_as_of_ts"])
    newest = adv["feature_as_of_ts"].max()
    cutoff = newest - pd.Timedelta(days=config.holdout_days)
    a = sorted(adv.loc[adv.feature_as_of_ts <= cutoff, "user_id"].unique())
    held = sorted(adv.loc[adv.feature_as_of_ts > cutoff, "user_id"].unique())
    return TwoStateSplit(cutoff=cutoff, newest=newest, state_a_users=a, holdout_users=held)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --group fraud pytest projects/fraud_anomaly_detection/codex_poc/tests/control/test_holdout.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add projects/fraud_anomaly_detection/codex_poc/control/holdout.py projects/fraud_anomaly_detection/codex_poc/tests/control/test_holdout.py
git commit -m "fraud control: two-state leak-free holdout split"
```

---

## Task 7: Monitoring stats

**Files:**
- Create: `codex_poc/control/monitoring.py`
- Test: `codex_poc/tests/control/test_monitoring.py`

Given a burned-key list (from State-A derivation) and the held-out users, compute the loop stats: prevention (held-out bad users a burned key would have caught), leakage (held-out discovered/bad users no key caught), and innocent capture in the holdout.

- [ ] **Step 1: Write the failing test**

```python
# codex_poc/tests/control/test_monitoring.py
import pandas as pd
from projects.fraud_anomaly_detection.codex_poc.control import monitoring


def test_holdout_effect_counts_prevention_and_leakage(tiny_store):
    # burned key = bank/acctA. Held-out users = u3 (acctA, DPD45) and u4 (devX, not bad).
    burned = pd.DataFrame({"entity_type": ["bank"], "entity_value": ["acctA"]})
    held = ["u3", "u4"]
    report = monitoring.holdout_effect(tiny_store, burned, held)
    assert report["prevented_bad"] == 1        # u3 caught by acctA, is DPD45
    assert report["innocents_blocked"] == 0     # u4 is on devX, not acctA
    assert report["leaked_bad"] == 0            # no held-out bad user left uncaught
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --group fraud pytest projects/fraud_anomaly_detection/codex_poc/tests/control/test_monitoring.py -v`
Expected: FAIL (cannot import monitoring.holdout_effect).

- [ ] **Step 3: Write minimal implementation**

```python
# codex_poc/control/monitoring.py
"""Monitoring stats over the holdout (CONTROL_SYSTEM_DESIGN §7).

Prevention / innocent-capture / leakage for a burned-key list applied to the
held-out users. Skeleton version: the daily live loop reuses this same code,
swapping the held-out set for 'yesterday' once production feeds back.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


def holdout_effect(store: Path | str, burned_keys: pd.DataFrame,
                   holdout_users: list[str]) -> dict:
    held = set(map(str, holdout_users))
    with duckdb.connect(str(store), read_only=True) as con:
        edges = con.execute(
            "SELECT entity_type, entity_value, CAST(user_id AS VARCHAR) AS user_id FROM edges").df()
        outcomes = con.execute(
            "SELECT CAST(user_id AS VARCHAR) AS user_id, "
            "max(CASE WHEN label_mature_d45 AND label_gross_dpd45 THEN 1 ELSE 0 END) AS bad, "
            "max(CASE WHEN label_mature_d45 THEN 1 ELSE 0 END) AS mature FROM advances GROUP BY 1"
        ).df()
    keyset = set(zip(burned_keys.entity_type, burned_keys.entity_value))
    caught = {u for et, ev, u in edges.itertuples(index=False) if (et, ev) in keyset}
    caught_held = caught & held
    out = outcomes.set_index("user_id")
    def _bad(u): return u in out.index and out.loc[u, "bad"] == 1
    held_bad = {u for u in held if _bad(u)}
    return {
        "holdout_users": len(held),
        "prevented_bad": len(caught_held & held_bad),
        "innocents_blocked": len({u for u in caught_held
                                  if u in out.index and out.loc[u, "mature"] == 1
                                  and out.loc[u, "bad"] == 0}),
        "leaked_bad": len(held_bad - caught_held),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --group fraud pytest projects/fraud_anomaly_detection/codex_poc/tests/control/test_monitoring.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add projects/fraud_anomaly_detection/codex_poc/control/monitoring.py projects/fraud_anomaly_detection/codex_poc/tests/control/test_monitoring.py
git commit -m "fraud control: holdout monitoring stats (prevention / leakage / innocents)"
```

---

## Task 8: End-to-end orchestrator + smoke test

**Files:**
- Create: `codex_poc/control/run.py`
- Test: `codex_poc/tests/control/test_run.py`

Wires the loop: run the registered discovery methods → write a finding snapshot → derive plugs on State-A discovered users → measure the holdout effect. This is the walking skeleton's "it all connects" proof.

- [ ] **Step 1: Write the failing test (real sample store)**

```python
# codex_poc/tests/control/test_run.py
from pathlib import Path
import pytest
from projects.fraud_anomaly_detection.codex_poc.control.config import ControlConfig
from projects.fraud_anomaly_detection.codex_poc.control.run import run_skeleton

SAMPLE = Path("projects/fraud_anomaly_detection/data/graph/fraud_graph.duckdb")


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample store not built")
def test_run_skeleton_end_to_end(tmp_path):
    report = run_skeleton(SAMPLE, findings_db=tmp_path / "f.duckdb",
                          config=ControlConfig(min_support=2, min_coverage=1,
                                               block_tier_precision=0.5))
    assert report["n_findings"] > 0           # discovery produced findings
    assert "burned_keys" in report            # plug derivation ran
    assert "holdout" in report                # monitoring ran
    assert set(report["holdout"]).issuperset({"prevented_bad", "leaked_bad"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --group fraud pytest projects/fraud_anomaly_detection/codex_poc/tests/control/test_run.py -v`
Expected: FAIL (cannot import run_skeleton).

- [ ] **Step 3: Write minimal implementation**

```python
# codex_poc/control/run.py
"""Walking-skeleton orchestrator: discovery -> finding store -> plugs -> holdout monitoring."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from projects.fraud_anomaly_detection.codex_poc.control import holdout, monitoring, plug
from projects.fraud_anomaly_detection.codex_poc.control.config import ControlConfig
from projects.fraud_anomaly_detection.codex_poc.control.discovery.graph_method import (
    ResidualRingMethod,
)
from projects.fraud_anomaly_detection.codex_poc.control.discovery.scenario_method import (
    ScenarioMethod,
)
from projects.fraud_anomaly_detection.codex_poc.control.finding_store import FindingStore

# The "representative few" wired for the skeleton — more methods plug in here.
METHODS = [ScenarioMethod("ring_account_reuse"), ResidualRingMethod()]


def run_skeleton(store: Path | str, findings_db: Path | str,
                 config: ControlConfig = ControlConfig(), refresh_key: str = "skeleton") -> dict:
    finding_sets = [m.run(store) for m in METHODS]
    fstore = FindingStore(findings_db)
    fstore.write_snapshot(refresh_key, data_version="sample", finding_sets=finding_sets)
    findings = fstore.read_latest()

    split = holdout.two_state_split(store, config)
    # derive on State A only: discovered users that exist in A (leak-free, P7)
    discovered_a = findings.loc[findings.user_id.isin(split.state_a_users), "user_id"]
    stats = plug.candidate_stats(store, discovered_a)
    burned = plug.qualify(stats, config)
    holdout_report = monitoring.holdout_effect(store, burned, split.holdout_users)
    return {
        "n_findings": int(findings.user_id.nunique()),
        "burned_keys": burned[["entity_type", "entity_value", "dpd45_precision",
                               "coverage", "support"]].to_dict("records"),
        "holdout": holdout_report,
    }
```

- [ ] **Step 4: Run test, the full control suite, verify pass**

Run: `uv run --group fraud pytest projects/fraud_anomaly_detection/codex_poc/tests/control/ -v`
Expected: PASS (all control tests; the two sample-store tests PASS or SKIP if no store).

- [ ] **Step 5: Commit**

```bash
git add projects/fraud_anomaly_detection/codex_poc/control/run.py projects/fraud_anomaly_detection/codex_poc/tests/control/test_run.py
git commit -m "fraud control: end-to-end walking-skeleton orchestrator + smoke test"
```

---

## Task 9: Sever the archived dependency (final cleanup)

**Files:**
- Modify: any `codex_poc/control/*.py` that imports from `codex_poc.archived`
- Test: `codex_poc/tests/control/test_no_archived_dep.py`

Policy (see `codex_poc/README.md`): borrowing from `archived/` *during*
development is fine, but the finished unit must carry **no dependency on
`archived/`** — copy any reused logic into `control/` and drop the import.
(Importing the live repo packages `scenarios` / `graph` / `analysis` is fine
and expected; this guard is only about `archived/`.)

- [ ] **Step 1: Copy out any reused archived logic**

If any module under `control/` imports from `...codex_poc.archived`, inline the
borrowed function/snippet into the appropriate `control/` module and remove the
import. If nothing was borrowed, this is a no-op — the guard below still locks it in.

- [ ] **Step 2: Write the guard test**

```python
# codex_poc/tests/control/test_no_archived_dep.py
import re
from pathlib import Path

CONTROL = Path("projects/fraud_anomaly_detection/codex_poc/control")


def test_control_has_no_archived_dependency():
    offenders = []
    for py in CONTROL.rglob("*.py"):
        if re.search(r"(from|import)\s+.*codex_poc\.archived", py.read_text()):
            offenders.append(str(py))
    assert not offenders, f"control/ must not depend on archived/: {offenders}"
```

- [ ] **Step 3: Run the guard, verify it passes**

Run: `uv run --group fraud pytest projects/fraud_anomaly_detection/codex_poc/tests/control/test_no_archived_dep.py -v`
Expected: PASS (no `control/` module imports `archived/`).

- [ ] **Step 4: Run the full control suite, verify green**

Run: `uv run --group fraud pytest projects/fraud_anomaly_detection/codex_poc/tests/control/ -v`
Expected: PASS (all control tests).

- [ ] **Step 5: Commit**

```bash
git add projects/fraud_anomaly_detection/codex_poc/control/ projects/fraud_anomaly_detection/codex_poc/tests/control/test_no_archived_dep.py
git commit -m "fraud control: sever archived dependency + guard test"
```

---

## Done — what the skeleton proves and what plugs in next

When all 9 tasks are green, the loop runs end-to-end on the sample: two real discovery methods (one scenario, one graph) emit findings through one contract → versioned finding store → plug extract/validate/qualify with config-driven thresholds → two-state leak-free holdout → monitoring stats. The contract and seams are proven.

**Plugs in later (no rebuild):** more discovery methods (one adapter each in `discovery/`, appended to `METHODS`); v3-tuned thresholds (edit `ControlConfig`); the plug lifecycle/expiry and the finding-snapshot trim history (extend `finding_store.py`/`plug.py`); the warehouse-facing burned-key table and live daily monitoring (swap the holdout set for "yesterday"). All gated on v3 data / VPN per `CONTROL_SYSTEM_DESIGN.md` §8.
