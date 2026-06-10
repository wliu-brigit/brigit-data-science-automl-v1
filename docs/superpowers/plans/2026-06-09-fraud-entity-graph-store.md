# Fraud Entity Graph Store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the persisted, queryable fraud entity graph from the approved spec (`docs/superpowers/specs/2026-06-09-fraud-entity-graph-store-design.md`): a self-contained DuckDB store + igraph analysis layer + demo on the 20k sample + DuckPGQ probe.

**Architecture:** Two decoupled layers. `graph/build.py` derives a lossless store (edges/users/entities/advances-snapshot/meta tables in one `.duckdb` file) from the base table via portable SQL. `graph/load.py` + `graph/queries.py` are parameterized views and question-level helpers on top (igraph engine). All judgment calls (caps, layers, as-of, scenario flags, weights) live in the analysis layer.

**Tech Stack:** DuckDB (store + SQL), python-igraph (analysis engine), existing scenario engine (`projects/fraud_anomaly_detection/scenarios/`), pytest (`pytest.mark.unit`, tiny synthetic frames, tmp_path).

**Context for a zero-context engineer:**
- Repo runs everything through `uv` (`uv run pytest ...`, `uv add ...`). Never pip.
- **Project-scoped dependencies (wendao decision, 2026-06-09):** packages needed
  only by this project live in a PEP 735 dependency group named `fraud`
  (`[dependency-groups]` in pyproject.toml — same mechanism as the existing
  `dev` group). Shared/base deps stay in `dev`. Anyone working this project
  runs `uv sync --group fraud` once (or uses `uv run --group fraud ...`
  per-command, as this plan's commands do). Everything still resolves in the
  one shared lock, so versions stay consistent across projects. Graph test
  files guard with `pytest.importorskip` so a bare `pytest` run stays green
  for sessions that never synced the group (skips, not failures).
- Project tests live in `projects/fraud_anomaly_detection/tests/` and are picked up by bare pytest (see `[tool.pytest.ini_options]`).
- The sample base table is `projects/fraud_anomaly_detection/data/sample/graph_sample.parquet` (20k advances × 110 cols; fraud-enriched; `*.parquet` is gitignored).
- Entity-key columns in the base table: `device_id`, `bank_account_key`, `persistent_account_id`, `phone_key`, `address_key`, `email_key`, `ip_address`. Advance id col: `advance_id`; user col: `user_id`; event time col: `feature_as_of_ts`.
- Scenario engine API: `from projects.fraud_anomaly_detection.scenarios import assign` (flags vs the bound register) and `from projects.fraud_anomaly_detection.scenarios.engine import load_register, evaluate` (custom register path — used in tests).
- Commit messages: prefix `fraud:`, end with the Claude co-author line (see git log for examples).

---

### Task 1: Dependencies, gitignore, package skeleton

**Files:**
- Modify: `pyproject.toml` (via `uv add`)
- Modify: `.gitignore`
- Create: `projects/fraud_anomaly_detection/graph/__init__.py`

- [x] **Step 1: Add dependencies to the project-scoped group**

Run: `uv add --group fraud duckdb igraph`
Expected: a new `fraud = ["duckdb>=...", "igraph>=..."]` entry appears under
`[dependency-groups]` in pyproject.toml (alongside `dev`), and both packages
install (igraph is the C-core graph library, PyPI name `igraph`).

- [x] **Step 2: Verify imports via the group**

Run: `uv run --group fraud python -c "import duckdb, igraph; print(duckdb.__version__, igraph.__version__)"`
Expected: two version strings, no error.

- [x] **Step 3: Gitignore the store files**

Append to `.gitignore` (after the `*.parquet` line):

```
*.duckdb
*.duckdb.wal
```

- [x] **Step 4: Create the package**

Create `projects/fraud_anomaly_detection/graph/__init__.py`:

```python
"""Persisted fraud entity graph — lossless DuckDB store + igraph analysis views.

Design: docs/superpowers/specs/2026-06-09-fraud-entity-graph-store-design.md
Store has no opinions (uncapped edges, full timestamps, all entity types,
self-contained advances snapshot); every judgment call — degree caps, layer
selection, as-of windows, scenario flags, weights — is a parameter of an
analysis-time view.

    from projects.fraud_anomaly_detection.graph import build, load, queries
"""
```

- [x] **Step 5: Document the group in the project README**

In `projects/fraud_anomaly_detection/README.md`, insert this section right
before `## Writing PROJECT_INSTRUCTIONS.md` (mirrors the scaffold template's
generic section, made concrete for this project):

```markdown
## Project-specific dependencies

Shared tooling lives in the repo's default `dev` dependency group. Packages
only this project needs live in the `fraud` dependency group (the same
`[dependency-groups]` mechanism as `dev`, one shared lockfile — consistent
versions repo-wide, opt-in install):

```bash
uv sync --group fraud                 # opt in, once per checkout
uv add --group fraud <package>        # add a new project dependency
uv run --group fraud python -m ...    # or per command, no sync needed
```

Current contents: `duckdb` + `igraph` — the persisted entity-graph store
(`graph/`). Tests importing group-only packages guard with
`pytest.importorskip`, so a bare `pytest` stays green without the group.
```

(Note: the inner ```bash fence ends before the section's final paragraph —
copy the structure exactly as shown.)

- [x] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore projects/fraud_anomaly_detection/graph/__init__.py \
        projects/fraud_anomaly_detection/README.md
git commit -m "fraud: graph package skeleton + project-scoped 'fraud' dep group (duckdb, igraph)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Shared test fixture (`conftest.py`)

The toy world every graph test uses. Hand-designed so every expected number in
later tasks is derivable by eye:

- Ring A (multi-type): `u1`–`u2` share device `d1`; `u2`–`u3` share bank `b1`.
  `u1` is fraud. `u2` has `loan_amount` 150 (toy-scenario trigger), the
  `u3`–`b1` edge is dated later (as-of tests).
- Singleton: `u4` on its own device `d4`.
- Hub: `u5`…`u9` (5 users) all on device `dH` (degree-cap tests).
- A literal duplicate source row (dedup test) and a sentinel `device_id="none"`
  row (screening test). One repeat advance `u1`–`d1` (parallel-edge test).

**Files:**
- Create: `projects/fraud_anomaly_detection/tests/conftest.py`

- [x] **Step 1: Write the fixture**

```python
"""Shared fixtures for the graph-store tests: a hand-checkable toy world."""

import pandas as pd
import pytest

_TS = pd.Timestamp


def _row(advance_id, user_id, ts, device=None, bank=None, persistent=None,
         phone=None, address=None, email=None, ip=None,
         loan_amount=50.0, is_fraud=0):
    return {
        "advance_id": advance_id, "user_id": user_id, "feature_as_of_ts": ts,
        "device_id": device, "bank_account_key": bank,
        "persistent_account_id": persistent, "phone_key": phone,
        "address_key": address, "email_key": email, "ip_address": ip,
        "loan_amount": loan_amount, "is_fraud": is_fraud,
        "label_gross_dpd45": is_fraud, "label_mature_d45": 1,
        "is_neobank_high_risk_institution": 0,
    }


@pytest.fixture()
def toy_df():
    rows = [
        _row("a01", "u1", _TS("2026-01-01 10:00"), device="d1", is_fraud=1),
        _row("a02", "u2", _TS("2026-01-02 10:00"), device="d1", loan_amount=150.0),
        _row("a02", "u2", _TS("2026-01-02 10:00"), device="d1", loan_amount=150.0),  # duplicate row
        _row("a03", "u2", _TS("2026-01-03 10:00"), bank="b1", loan_amount=150.0),
        _row("a04", "u3", _TS("2026-02-01 10:00"), bank="b1"),  # late edge (as-of tests)
        _row("a05", "u4", _TS("2026-01-04 10:00"), device="d4"),
        _row("a06", "u5", _TS("2026-01-05 10:00"), device="dH"),
        _row("a07", "u6", _TS("2026-01-06 10:00"), device="dH"),
        _row("a08", "u7", _TS("2026-01-07 10:00"), device="dH"),
        _row("a09", "u8", _TS("2026-01-08 10:00"), device="dH"),
        _row("a10", "u9", _TS("2026-01-09 10:00"), device="dH"),
        _row("a11", "u1", _TS("2026-01-06 11:00"), device="d1"),  # parallel edge u1-d1
        _row("a12", "u1", _TS("2026-01-07 11:00"), device="none"),  # sentinel -> screened
    ]
    return pd.DataFrame(rows)


@pytest.fixture()
def toy_store(toy_df, tmp_path):
    from projects.fraud_anomaly_detection.graph.build import build_store

    path = tmp_path / "toy_graph.duckdb"
    build_store(toy_df, path, source_label="toy")
    return path
```

Expected toy facts used throughout (derive once, reuse):
- device edges after screening+dedup: `(a01,u1,d1) (a02,u2,d1) (a05,u4,d4) (a06..a10 → 5×dH) (a11,u1,d1)` = **9**
- bank edges: `(a03,u2,b1) (a04,u3,b1)` = **2**; all other types = **0**
- screened device cells = **1** (`a12`), distinct users = **9**, advances = **12**
- `d1` has 2 distinct users, `dH` has 5, `b1` has 2, `d4` has 1.

- [x] **Step 2: Commit**

```bash
git add projects/fraud_anomaly_detection/tests/conftest.py
git commit -m "fraud: toy-world fixture for graph-store tests

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: The store builder (`graph/build.py`)

**Files:**
- Create: `projects/fraud_anomaly_detection/graph/build.py`
- Test: `projects/fraud_anomaly_detection/tests/test_graph_build.py`

- [x] **Step 1: Write the failing tests**

`projects/fraud_anomaly_detection/tests/test_graph_build.py`:

```python
"""Store builder: lossless edges, sentinel screening, self-contained snapshot."""

import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")  # project deps: uv sync --group fraud

from projects.fraud_anomaly_detection.graph.build import build_store  # noqa: E402

pytestmark = pytest.mark.unit


def _q(path, sql):
    with duckdb.connect(str(path), read_only=True) as con:
        return con.execute(sql).fetchall()


def test_edges_lossless_per_type(toy_store):
    counts = dict(_q(toy_store, "SELECT entity_type, count(*) FROM edges GROUP BY 1"))
    assert counts == {"device": 9, "bank": 2}


def test_no_degree_cap_applied(toy_store):
    [(n_users,)] = _q(toy_store,
        "SELECT n_users FROM entities WHERE entity_value = 'dH'")
    assert n_users == 5  # hub kept in full — caps are analysis-time only


def test_sentinels_screened_and_counted(toy_store):
    assert _q(toy_store, "SELECT count(*) FROM edges WHERE entity_value = 'none'") == [(0,)]
    [(v,)] = _q(toy_store, "SELECT value FROM meta WHERE key = 'screened_device'")
    assert int(v) == 1


def test_duplicate_source_rows_dedup_to_one_edge(toy_store):
    assert _q(toy_store,
        "SELECT count(*) FROM edges WHERE advance_id = 'a02'") == [(1,)]


def test_advances_snapshot_is_complete(toy_store, toy_df):
    [(n,)] = _q(toy_store, "SELECT count(*) FROM advances")
    assert n == len(toy_df)  # full snapshot, duplicates and sentinels included
    cols = [r[0] for r in _q(toy_store, "DESCRIBE advances")]
    assert set(cols) == set(toy_df.columns)


def test_aggregates_consistent_with_edges(toy_store):
    mismatches = _q(toy_store, """
        SELECT e.entity_value FROM entities e JOIN (
            SELECT entity_type, entity_value, count(DISTINCT user_id) nu
            FROM edges GROUP BY 1, 2) c
        USING (entity_type, entity_value) WHERE e.n_users <> c.nu""")
    assert mismatches == []
    [(n_users,)] = _q(toy_store, "SELECT count(*) FROM users")
    assert n_users == 9


def test_rebuild_is_idempotent(toy_df, tmp_path):
    path = tmp_path / "g.duckdb"
    first = build_store(toy_df, path, source_label="toy")
    second = build_store(toy_df, path, source_label="toy")
    assert first == second
    assert _q(path, "SELECT count(*) FROM edges") == [(11,)]
```

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run --group fraud pytest projects/fraud_anomaly_detection/tests/test_graph_build.py -q`
Expected: collection error — `No module named 'projects.fraud_anomaly_detection.graph.build'`

- [x] **Step 3: Implement `graph/build.py`**

```python
"""Build the persisted entity-graph store: one lossless, self-contained DuckDB file.

The build is a pure SQL transformation of the base table (kept
warehouse-portable: plain SELECT/UNION ALL — the same shape can run in
Snowflake). No judgment calls here: edges are UNCAPPED, every entity type is
stored (ip/email included; analysis views exclude them by default), and the
full base table is snapshotted in (`advances`) so the file alone carries all
metadata and labels. The only build-time cleaning is sentinel screening —
non-values like '' / 'none' / '0-0' are not identities — with screened counts
logged to `meta`.

Refresh model: full rebuild only (idempotent — the file is replaced).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

# entity_type -> base-table column. THE canonical map; load.py reuses it.
ENTITY_COLS: dict[str, str] = {
    "device": "device_id",
    "bank": "bank_account_key",
    "persistent": "persistent_account_id",
    "phone": "phone_key",
    "address": "address_key",
    "email": "email_key",
    "ip": "ip_address",
}

# Non-values seen in the wild (prior graph effort + v2 due diligence).
SENTINELS: tuple[str, ...] = ("", "none", "nan", "null", "nat", "0", "0-0", "none-none")

ADVANCE_ID, USER_ID, TS = "advance_id", "user_id", "feature_as_of_ts"


def _sentinel_list() -> str:
    return ", ".join(f"'{s}'" for s in SENTINELS)


def _edge_select(etype: str, col: str) -> str:
    value = f"lower(trim(CAST({col} AS VARCHAR)))"
    return f"""
        SELECT DISTINCT {ADVANCE_ID} AS advance_id, {USER_ID} AS user_id,
               '{etype}' AS entity_type,
               CAST({col} AS VARCHAR) AS entity_value, {TS} AS ts
        FROM advances
        WHERE {col} IS NOT NULL AND {value} NOT IN ({_sentinel_list()})
    """


def build_store(
    source: Path | str | pd.DataFrame,
    out_path: Path | str,
    source_label: str = "",
) -> dict[str, int]:
    """Derive the store from the base table; returns the count summary.

    `source` is a DataFrame or a parquet path. The output file is replaced
    (full-rebuild refresh model). Summary counts are also persisted to `meta`.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    Path(f"{out}.wal").unlink(missing_ok=True)

    con = duckdb.connect(str(out))
    try:
        if isinstance(source, pd.DataFrame):
            con.register("src", source)
            con.execute("CREATE TABLE advances AS SELECT * FROM src")
        else:
            con.execute(
                "CREATE TABLE advances AS SELECT * FROM read_parquet(?)", [str(source)]
            )

        edge_union = "\nUNION ALL\n".join(
            _edge_select(etype, col) for etype, col in ENTITY_COLS.items()
        )
        con.execute(f"CREATE TABLE edges AS {edge_union}")

        con.execute(f"""
            CREATE TABLE users AS
            SELECT {USER_ID} AS user_id, count(DISTINCT {ADVANCE_ID}) AS n_advances,
                   min({TS}) AS first_seen_ts, max({TS}) AS last_seen_ts
            FROM advances GROUP BY 1
        """)
        con.execute("""
            CREATE TABLE entities AS
            SELECT entity_type, entity_value,
                   count(DISTINCT user_id) AS n_users,
                   count(DISTINCT advance_id) AS n_advances,
                   min(ts) AS first_seen_ts, max(ts) AS last_seen_ts
            FROM edges GROUP BY 1, 2
        """)

        summary: dict[str, int] = {}
        summary["n_advances"] = con.execute("SELECT count(*) FROM advances").fetchone()[0]
        summary["n_users"] = con.execute("SELECT count(*) FROM users").fetchone()[0]
        summary["n_edges"] = con.execute("SELECT count(*) FROM edges").fetchone()[0]
        for etype, col in ENTITY_COLS.items():
            summary[f"edges_{etype}"] = con.execute(
                "SELECT count(*) FROM edges WHERE entity_type = ?", [etype]
            ).fetchone()[0]
            value = f"lower(trim(CAST({col} AS VARCHAR)))"
            summary[f"screened_{etype}"] = con.execute(
                f"SELECT count(*) FROM advances WHERE {col} IS NOT NULL "
                f"AND {value} IN ({_sentinel_list()})"
            ).fetchone()[0]

        con.execute("CREATE TABLE meta (key VARCHAR, value VARCHAR)")
        con.execute("INSERT INTO meta VALUES ('built_at', CAST(current_timestamp AS VARCHAR))")
        con.execute("INSERT INTO meta VALUES ('source', ?)", [source_label or str(source)])
        for key, val in summary.items():
            con.execute("INSERT INTO meta VALUES (?, ?)", [key, str(val)])
        return summary
    finally:
        con.close()
```

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run --group fraud pytest projects/fraud_anomaly_detection/tests/test_graph_build.py -q`
Expected: 7 passed.

- [x] **Step 5: Commit**

```bash
git add projects/fraud_anomaly_detection/graph/build.py \
        projects/fraud_anomaly_detection/tests/test_graph_build.py
git commit -m "fraud: graph store builder — lossless edges + self-contained snapshot

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Parameterized views (`graph/load.py`)

**Files:**
- Create: `projects/fraud_anomaly_detection/graph/load.py`
- Test: `projects/fraud_anomaly_detection/tests/test_graph_load.py`

- [x] **Step 1: Write the failing tests**

`projects/fraud_anomaly_detection/tests/test_graph_load.py`:

```python
"""Analysis views: layers, caps, as-of, metadata join, dynamic scenario overlay."""

import textwrap

import pandas as pd
import pytest

pytest.importorskip("duckdb")  # project deps: uv sync --group fraud
pytest.importorskip("igraph")

from projects.fraud_anomaly_detection.graph.load import load_graph  # noqa: E402

pytestmark = pytest.mark.unit

TOY_REGISTER = """\
version: "test.1"
scenarios:
  - name: big_loan
    tier: review
    status: draft
    entry_date: 2026-06-09
    theory: toy
    trigger:
      - column: loan_amount
        op: ">"
        value: 100
"""


def _users(g):
    return {v["raw_id"] for v in g.vs if v["kind"] == "user"}


def _entities(g):
    return {v["raw_id"] for v in g.vs if v["kind"] != "user"}


def test_default_load_shape(toy_store):
    g = load_graph(toy_store, scenarios=False)
    assert _users(g) == {f"u{i}" for i in range(1, 10)}  # ALL users, singletons too
    assert _entities(g) == {"d1", "d4", "dH", "b1"}
    assert g.ecount() == 11


def test_layer_selection(toy_store):
    g = load_graph(toy_store, layers=("bank",), scenarios=False)
    assert _entities(g) == {"b1"}
    assert g.ecount() == 2


def test_unknown_layer_raises(toy_store):
    with pytest.raises(ValueError, match="unknown layer"):
        load_graph(toy_store, layers=("device", "passport"), scenarios=False)


def test_degree_cap_drops_hub_keeps_users(toy_store):
    g = load_graph(toy_store, degree_cap=4, scenarios=False)
    assert "dH" not in _entities(g)
    assert {"u5", "u6", "u7", "u8", "u9"} <= _users(g)  # users stay (as singletons)


def test_as_of_excludes_future_edges(toy_store):
    g = load_graph(toy_store, as_of=pd.Timestamp("2026-01-15"), scenarios=False)
    etypes = {e["etype"] for e in g.es}
    assert "bank" in etypes
    bank_edges = [e for e in g.es if e["etype"] == "bank"]
    assert len(bank_edges) == 1  # a04 (2026-02-01) excluded


def test_window_slices_edges(toy_store):
    g = load_graph(
        toy_store, layers=("device",),
        window=(pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-09 23:00")),
        scenarios=False,
    )
    assert g.ecount() == 6  # hub edges a06..a10 + the a11 parallel edge


def test_parallel_edges_kept(toy_store):
    g = load_graph(toy_store, scenarios=False)
    u1, d1 = g.vs.find(name="user:u1"), g.vs.find(name="device:d1")
    assert len(g.es.select(_between=([u1.index], [d1.index]))) == 2


def test_node_attrs_joined_from_snapshot(toy_store):
    g = load_graph(toy_store, node_attrs=("is_fraud",), scenarios=False)
    assert g.vs.find(name="user:u1")["is_fraud"] == 1
    assert g.vs.find(name="user:u2")["is_fraud"] == 0


def test_scenario_overlay_matches_engine(toy_store, tmp_path):
    register = tmp_path / "register.yaml"
    register.write_text(textwrap.dedent(TOY_REGISTER))
    g = load_graph(toy_store, register_path=register)
    flagged = {v["raw_id"] for v in g.vs if v["kind"] == "user" and v["scenario_big_loan"]}
    assert flagged == {"u2"}  # only u2 has loan_amount > 100
    assert g.vs.find(name="user:u2")["scenario_any"]
```

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run --group fraud pytest projects/fraud_anomaly_detection/tests/test_graph_load.py -q`
Expected: collection error — `No module named '...graph.load'`

- [x] **Step 3: Implement `graph/load.py`**

```python
"""Load parameterized graph views from the store.

The store is lossless; THIS is where opinions are applied — which layers,
which degree cap, which time slice, which metadata, which scenario register.
Returns an igraph bipartite multigraph: user vertices (kind='user') and
entity vertices (kind=entity_type), parallel edges kept with etype/ts attrs.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import igraph as ig
import pandas as pd

from projects.fraud_anomaly_detection.graph.build import ENTITY_COLS, USER_ID

# email is near-noise (max 6 users even on full v3) and raw IP is a
# NAT/household junk generator (v1 learning) — stored, but opt-in.
DEFAULT_LAYERS: tuple[str, ...] = ("device", "bank", "persistent", "phone", "address")
DEFAULT_NODE_ATTRS: tuple[str, ...] = (
    "is_fraud", "label_gross_dpd45", "label_mature_d45",
    "is_neobank_high_risk_institution",
)


def _read_store(store: Path | str, sql: str, params: list | None = None) -> pd.DataFrame:
    with duckdb.connect(str(store), read_only=True) as con:
        return con.execute(sql, params or []).df()


def load_graph(
    store: Path | str,
    base: pd.DataFrame | None = None,
    layers: tuple[str, ...] = DEFAULT_LAYERS,
    degree_cap: int | None = None,
    as_of: pd.Timestamp | None = None,
    window: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    node_attrs: tuple[str, ...] = DEFAULT_NODE_ATTRS,
    scenarios: bool = True,
    register_path: Path | str | None = None,
) -> ig.Graph:
    """One opinionated view of the stored graph, as an igraph multigraph.

    base defaults to the store's own `advances` snapshot (the file is
    self-contained); pass a DataFrame to override. degree_cap drops entity
    vertices whose distinct-user count WITHIN THIS VIEW exceeds the cap
    (users always stay). scenarios=True runs the register (the bound one, or
    `register_path`) against `base` NOW and attaches user-level
    scenario_<name> / scenario_any flags — never persisted, always current.
    """
    unknown = set(layers) - set(ENTITY_COLS)
    if unknown:
        raise ValueError(f"unknown layer(s) {sorted(unknown)}; expected {sorted(ENTITY_COLS)}")

    conds = ["entity_type IN (" + ", ".join("?" * len(layers)) + ")"]
    params: list = list(layers)
    if as_of is not None:
        conds.append("ts <= ?")
        params.append(as_of)
    if window is not None:
        conds.append("ts >= ? AND ts <= ?")
        params.extend([window[0], window[1]])
    where = " AND ".join(conds)

    cap_filter = ""
    if degree_cap is not None:
        cap_filter = (
            " QUALIFY count(DISTINCT user_id) OVER "
            "(PARTITION BY entity_type, entity_value) <= ?"
        )
        params.append(degree_cap)

    edges = _read_store(
        store,
        f"SELECT advance_id, user_id, entity_type, entity_value, ts "
        f"FROM edges WHERE {where}{cap_filter}",
        params,
    )
    users = _read_store(store, "SELECT user_id FROM users ORDER BY 1")
    if base is None:
        base = _read_store(store, "SELECT * FROM advances")

    user_names = ("user:" + users[USER_ID].astype(str)).tolist()
    ent_ids = edges["entity_type"] + ":" + edges["entity_value"]
    ent_names = sorted(set(ent_ids))

    g = ig.Graph()
    g.add_vertices(user_names + ent_names)
    g.vs["kind"] = ["user"] * len(user_names) + [n.split(":", 1)[0] for n in ent_names]
    g.vs["raw_id"] = [n.split(":", 1)[1] for n in user_names + ent_names]

    index = {name: i for i, name in enumerate(g.vs["name"])}
    src = ("user:" + edges["user_id"].astype(str)).map(index)
    dst = ent_ids.map(index)
    g.add_edges(list(zip(src, dst)))
    g.es["etype"] = edges["entity_type"].tolist()
    g.es["ts"] = list(edges["ts"])

    flags = pd.DataFrame(index=base.index)
    if scenarios:
        if register_path is not None:
            from projects.fraud_anomaly_detection.scenarios import engine

            register = engine.load_register(register_path)
            flags = engine.evaluate(base, register.scenarios)
        else:
            from projects.fraud_anomaly_detection.scenarios import assign

            flags = assign(base)

    per_user = pd.concat([base[[USER_ID]], base[list(node_attrs)], flags], axis=1)
    agg = per_user.groupby(USER_ID).max()  # labels/flags: any advance counts
    for col in agg.columns:
        values = agg[col].reindex([n.split(":", 1)[1] for n in user_names])
        g.vs[: len(user_names)][col] = values.tolist()
    return g
```

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run --group fraud pytest projects/fraud_anomaly_detection/tests/test_graph_load.py -q`
Expected: 9 passed. (If `QUALIFY` placement errors: move the cap into a
subquery — `SELECT * FROM (SELECT ..., count(DISTINCT user_id) OVER (...) AS du
FROM edges WHERE ...) WHERE du <= ?` — same semantics.)

- [x] **Step 5: Run build tests too (shared fixture untouched?)**

Run: `uv run --group fraud pytest projects/fraud_anomaly_detection/tests/test_graph_build.py projects/fraud_anomaly_detection/tests/test_graph_load.py -q`
Expected: all pass.

- [x] **Step 6: Commit**

```bash
git add projects/fraud_anomaly_detection/graph/load.py \
        projects/fraud_anomaly_detection/tests/test_graph_load.py
git commit -m "fraud: parameterized graph views — layers, caps, as-of, scenario overlay

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Question-level helpers (`graph/queries.py`)

**Files:**
- Create: `projects/fraud_anomaly_detection/graph/queries.py`
- Test: `projects/fraud_anomaly_detection/tests/test_graph_queries.py`

- [x] **Step 1: Write the failing tests**

`projects/fraud_anomaly_detection/tests/test_graph_queries.py`:

```python
"""Query helpers: proximity, components, rings, projection, hub report."""

import pytest

pytest.importorskip("duckdb")  # project deps: uv sync --group fraud
pytest.importorskip("igraph")

from projects.fraud_anomaly_detection.graph.load import load_graph  # noqa: E402
from projects.fraud_anomaly_detection.graph.queries import (  # noqa: E402
    components,
    hub_report,
    near_flagged,
    project_users,
    ring,
)

pytestmark = pytest.mark.unit


@pytest.fixture()
def toy_graph(toy_store):
    return load_graph(toy_store, node_attrs=("is_fraud",), scenarios=False)


def test_near_flagged_user_hops(toy_graph):
    out = near_flagged(toy_graph, flag="is_fraud", max_hops=3)
    hops = dict(zip(out["user_id"], out["hops"]))
    # u1 is the only fraud seed. u2 shares d1 (1 user-hop); u3 reaches u1
    # via b1-u2-d1 (2 user-hops). u4..u9 are unreachable -> absent.
    assert hops == {"u2": 1, "u3": 2}
    assert dict(zip(out["user_id"], out["nearest_flagged"])) == {"u2": "u1", "u3": "u1"}


def test_near_flagged_respects_max_hops(toy_graph):
    out = near_flagged(toy_graph, flag="is_fraud", max_hops=1)
    assert set(out["user_id"]) == {"u2"}


def test_components_census(toy_graph):
    out = components(toy_graph, flag="is_fraud")
    by_size = out.sort_values("n_users", ascending=False).reset_index(drop=True)
    assert by_size.loc[0, "n_users"] == 5            # hub component (dH)
    assert by_size.loc[0, "n_types"] == 1
    ring_row = by_size[by_size["n_users"] == 3].iloc[0]  # u1-u2-u3 via d1+b1
    assert ring_row["n_types"] == 2                  # multi-type: the v1 discriminator
    assert ring_row["n_flagged"] == 1                # u1
    assert (by_size["n_users"] == 1).sum() == 1      # u4+d4


def test_ring_extraction(toy_graph):
    sub = ring(toy_graph, "u1", hops=1)
    names = set(sub.vs["name"])
    assert names == {"user:u1", "device:d1", "user:u2"}


def test_project_users_weights(toy_store):
    out = project_users(toy_store, degree_cap=None)
    pairs = {tuple(sorted((a, b))): (s, t) for a, b, s, t
             in zip(out["user_a"], out["user_b"], out["n_shared"], out["n_types"])}
    assert pairs[("u1", "u2")] == (1, 1)   # share d1 only
    assert pairs[("u2", "u3")] == (1, 1)   # share b1 only
    assert len([p for p in pairs if "u5" in p or "u6" in p]) == 4 + 3  # hub pairs for u5/u6


def test_project_users_cap_kills_hub_pairs(toy_store):
    out = project_users(toy_store, degree_cap=4)
    users_in_pairs = set(out["user_a"]) | set(out["user_b"])
    assert users_in_pairs == {"u1", "u2", "u3"}


def test_hub_report_orders_and_annotates(toy_store):
    out = hub_report(toy_store, top_n=3)
    top = out.iloc[0]
    assert (top["entity_value"], top["n_users"]) == ("dH", 5)
    assert top["fraud_user_rate"] == 0.0
    d1 = out[out["entity_value"] == "d1"].iloc[0]
    assert d1["fraud_user_rate"] == 0.5  # u1 fraud, u2 not
```

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run --group fraud pytest projects/fraud_anomaly_detection/tests/test_graph_queries.py -q`
Expected: collection error — `No module named '...graph.queries'`

- [x] **Step 3: Implement `graph/queries.py`**

```python
"""Question-level helpers on a loaded graph view (or directly on the store).

This is the query surface (the engine has no query language): each function
answers one analysis question. Graph inputs are bipartite user<->entity views
from load.load_graph; `hops` always means USER-hops (2 bipartite steps).
project_users / hub_report run as SQL on the store: set math is the
database's home turf, traversal is the graph's.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import igraph as ig
import pandas as pd

from projects.fraud_anomaly_detection.graph.build import ENTITY_COLS
from projects.fraud_anomaly_detection.graph.load import DEFAULT_LAYERS


def _flagged_indices(g: ig.Graph, flag: str) -> list[int]:
    return [v.index for v in g.vs if v["kind"] == "user" and bool(v[flag])]


def near_flagged(g: ig.Graph, flag: str = "is_fraud", max_hops: int = 3) -> pd.DataFrame:
    """Users within max_hops USER-hops of any flagged user (flagged excluded).

    Single multi-source BFS via a virtual vertex attached to every flagged
    user: distance(virtual -> target) = 1 + bipartite-steps, and one
    user-hop = 2 bipartite steps, so hops = (d - 1) // 2.
    """
    seeds = _flagged_indices(g, flag)
    if not seeds:
        return pd.DataFrame(columns=["user_id", "hops", "nearest_flagged"])
    seed_set = set(seeds)
    gv = g.copy()
    virtual = gv.add_vertex(name="__virtual__", kind="__virtual__")
    gv.add_edges([(virtual.index, s) for s in seeds])
    dist = gv.distances(source=[virtual.index])[0]

    rows = []
    for v in g.vs:
        if v["kind"] != "user" or v.index in seed_set:
            continue
        d = dist[v.index]
        hops = (int(d) - 1) // 2 if d != float("inf") else None
        if hops is not None and 1 <= hops <= max_hops:
            path = gv.get_shortest_paths(virtual.index, to=v.index)[0]
            rows.append((v["raw_id"], hops, gv.vs[path[1]]["raw_id"]))
    return pd.DataFrame(rows, columns=["user_id", "hops", "nearest_flagged"])


def components(g: ig.Graph, flag: str = "is_fraud") -> pd.DataFrame:
    """Connected-component census with the multi-type density discriminator.

    n_types counts distinct ENTITY types in the component — the v1 finding:
    small components webbed across >=2 types are the ring signature.
    """
    comps = g.connected_components()
    rows = []
    for comp_id, members in enumerate(comps):
        kinds = [g.vs[i]["kind"] for i in members]
        users = [g.vs[i] for i, k in zip(members, kinds) if k == "user"]
        etypes = sorted({k for k in kinds if k != "user"})
        rows.append({
            "comp_id": comp_id,
            "n_users": len(users),
            "n_entities": len(members) - len(users),
            "entity_types": ",".join(etypes),
            "n_types": len(etypes),
            "n_flagged": sum(bool(v[flag]) for v in users),
            "user_ids": ",".join(sorted(v["raw_id"] for v in users)),
        })
    return pd.DataFrame(rows)


def ring(g: ig.Graph, user_id: str, hops: int = 2) -> ig.Graph:
    """Ego subgraph around a user, out to `hops` user-hops (deep-dive unit)."""
    center = g.vs.find(name=f"user:{user_id}")
    member_ids = g.neighborhood(center.index, order=2 * hops)
    return g.induced_subgraph(member_ids)


def project_users(
    store: Path | str,
    layers: tuple[str, ...] = DEFAULT_LAYERS,
    degree_cap: int | None = 20,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Weighted user<->user projection: n_shared entities + n_types distinct types.

    ALWAYS think before lifting the cap: one 136-user device alone emits
    9,180 pairs. Default cap matches the v1 ring-traversal finding (~20).
    """
    unknown = set(layers) - set(ENTITY_COLS)
    if unknown:
        raise ValueError(f"unknown layer(s) {sorted(unknown)}; expected {sorted(ENTITY_COLS)}")
    params: list = list(layers)
    time_cond = ""
    if as_of is not None:
        time_cond = " AND ts <= ?"
        params.append(as_of)
    cap_cond = ""
    if degree_cap is not None:
        cap_cond = (
            " AND (entity_type, entity_value) IN ("
            "SELECT entity_type, entity_value FROM entities WHERE n_users <= ?)"
        )
        params.append(degree_cap)

    sql = f"""
        WITH pairs AS (
            SELECT DISTINCT user_id, entity_type, entity_value FROM edges
            WHERE entity_type IN ({", ".join("?" * len(layers))}){time_cond}{cap_cond}
        )
        SELECT a.user_id AS user_a, b.user_id AS user_b,
               count(*) AS n_shared,
               count(DISTINCT a.entity_type) AS n_types
        FROM pairs a JOIN pairs b
          ON a.entity_type = b.entity_type AND a.entity_value = b.entity_value
         AND a.user_id < b.user_id
        GROUP BY 1, 2
    """
    with duckdb.connect(str(store), read_only=True) as con:
        return con.execute(sql, params).df()


def hub_report(
    store: Path | str,
    top_n: int = 20,
    layers: tuple[str, ...] = tuple(ENTITY_COLS),
) -> pd.DataFrame:
    """High-degree entities, NO cap — the bigger, the more interesting.

    The time axis separates fraud farms (many users in days, high attached
    fraud rate) from shared infrastructure (many users over years, base rate).
    """
    params: list = list(layers)
    sql = f"""
        WITH user_label AS (
            SELECT user_id, max(is_fraud) AS is_fraud FROM advances GROUP BY 1
        ), pairs AS (
            SELECT DISTINCT entity_type, entity_value, user_id FROM edges
            WHERE entity_type IN ({", ".join("?" * len(layers))})
        ), stats AS (
            SELECT entity_type, entity_value,
                   count(DISTINCT user_id) AS n_users,
                   count(*) AS n_edges,
                   date_diff('day', min(ts), max(ts)) AS span_days
            FROM edges
            WHERE entity_type IN ({", ".join("?" * len(layers))})
            GROUP BY 1, 2
        )
        SELECT s.*, round(s.n_users / greatest(s.span_days, 1), 3) AS users_per_day,
               round(avg(l.is_fraud), 3) AS fraud_user_rate
        FROM stats s
        JOIN pairs p USING (entity_type, entity_value)
        JOIN user_label l USING (user_id)
        GROUP BY ALL
        ORDER BY s.n_users DESC, s.entity_value
        LIMIT {int(top_n)}
    """
    with duckdb.connect(str(store), read_only=True) as con:
        return con.execute(sql, params + params).df()
```

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run --group fraud pytest projects/fraud_anomaly_detection/tests/test_graph_queries.py -q`
Expected: 7 passed. (If the `(a, b) IN (SELECT (a, b) ...)` row-constructor
form errors on this DuckDB version, rewrite cap_cond as a join against
`entities` filtered on `n_users <= ?` — same semantics.)

- [x] **Step 5: Commit**

```bash
git add projects/fraud_anomaly_detection/graph/queries.py \
        projects/fraud_anomaly_detection/tests/test_graph_queries.py
git commit -m "fraud: graph query helpers — proximity, components, rings, projection, hubs

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Demo on the sample (`analysis/graph_store_demo.py`)

The session's acceptance test: build → persist → reopen → query, every
spec question answered with real output. NOT a precision study (sample
metrics/structure don't transfer — spec risks 1–2).

**Files:**
- Create: `projects/fraud_anomaly_detection/analysis/graph_store_demo.py`

- [x] **Step 1: Write the demo script**

```python
"""Capability demo for the persisted entity-graph store, on the local sample.

Proves the spec's question list end-to-end: build -> persist -> reopen ->
flag (current register, dynamically) -> proximity / per-layer / components /
hubs / ring deep-dive. Read-only toward the sample; writes only the store
file. Sample is fraud-enriched and graph-thinned: NUMBERS HERE ARE
CAPABILITY EVIDENCE, NOT TRANSFERABLE METRICS.

    uv run --group fraud python -m projects.fraud_anomaly_detection.analysis.graph_store_demo
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from projects.fraud_anomaly_detection.graph.build import build_store
from projects.fraud_anomaly_detection.graph.load import DEFAULT_LAYERS, load_graph
from projects.fraud_anomaly_detection.graph.queries import (
    components,
    hub_report,
    near_flagged,
    project_users,
    ring,
)

PROJECT = Path("projects/fraud_anomaly_detection")
SAMPLE = PROJECT / "data" / "sample" / "graph_sample.parquet"
STORE = PROJECT / "data" / "graph" / "fraud_graph.duckdb"


def banner(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    banner("1) BUILD: sample parquet -> persisted store")
    summary = build_store(SAMPLE, STORE, source_label=f"sample:{SAMPLE.name}")
    for key, val in summary.items():
        print(f"  {key:<22}{val:>10,}")

    banner("2) REOPEN: fresh connection, plain SQL inspection")
    with duckdb.connect(str(STORE), read_only=True) as con:
        print(con.execute(
            "SELECT entity_type, count(*) n_edges, count(DISTINCT entity_value) n_values"
            " FROM edges GROUP BY 1 ORDER BY 2 DESC").df().to_string(index=False))

    banner("3) LOAD + dynamic scenario overlay (current register)")
    base = pd.read_parquet(SAMPLE)
    from projects.fraud_anomaly_detection.scenarios import TRIGGER_COLUMNS

    missing = sorted(set(TRIGGER_COLUMNS) - set(base.columns))
    if missing:
        raise SystemExit(f"sample is missing register trigger columns: {missing}")
    g = load_graph(STORE, base=base)
    flag_cols = sorted(a for a in g.vs.attributes() if a.startswith("scenario_"))
    for col in flag_cols:
        n = sum(bool(v[col]) for v in g.vs if v["kind"] == "user")
        print(f"  {col:<40}{n:>8,} users")

    banner("4) PROXIMITY: users within 1-3 user-hops of a scenario-flagged user")
    out = near_flagged(g, flag="scenario_any", max_hops=3)
    print(f"  union graph ({'+'.join(DEFAULT_LAYERS)}): {len(out):,} users near a flagged one")
    print(out["hops"].value_counts().sort_index().rename("users").to_string())
    for layer in DEFAULT_LAYERS:
        gl = load_graph(STORE, base=base, layers=(layer,))
        nl = len(near_flagged(gl, flag="scenario_any", max_hops=3))
        print(f"  {layer:>12}-only: {nl:,}")

    banner("5) COMPONENT CENSUS: multi-type density (the v1 discriminator)")
    cap_g = load_graph(STORE, base=base, degree_cap=20)
    cc = components(cap_g, flag="is_fraud")
    multi = cc[(cc.n_users >= 3) & (cc.n_types >= 2)]
    print(f"  components (cap=20): {len(cc):,}; with >=3 users & >=2 types: {len(multi):,}")
    print(f"  fraud users inside multi-type comps: {int(multi.n_flagged.sum()):,}"
          f" / {int(multi.n_users.sum()):,} members")
    print(multi.sort_values("n_users", ascending=False).head(10)
          [["n_users", "n_types", "entity_types", "n_flagged"]].to_string(index=False))

    banner("6) HUB REPORT: no cap — farms vs infrastructure by time density")
    print(hub_report(STORE, top_n=15).to_string(index=False))

    banner("7) RING DEEP-DIVE: largest multi-type component")
    if len(multi):
        target_users = multi.sort_values("n_users", ascending=False).iloc[0]["user_ids"]
        center = target_users.split(",")[0]
        sub = ring(cap_g, center, hops=2)
        print(f"  ego graph around {center}: {sub.vcount()} vertices, {sub.ecount()} edges")
        proj = project_users(STORE, degree_cap=20)
        strong = proj[proj.n_types >= 2]
        print(f"  user-user projected pairs (cap=20): {len(proj):,};"
              f" multi-type pairs: {len(strong):,}")
    else:
        print("  no multi-type component >=3 users in the (thinned) sample")

    print("\nStore persisted at:", STORE)


if __name__ == "__main__":
    main()
```

- [x] **Step 2: Run the demo**

Run: `uv run --group fraud python -m projects.fraud_anomaly_detection.analysis.graph_store_demo`
Expected: all 7 banners print with non-trivial numbers; store file exists at
`projects/fraud_anomaly_detection/data/graph/fraud_graph.duckdb`; second run
succeeds identically (rebuild idempotent). If step 3 aborts on missing
trigger columns, STOP and surface to wendao (the sample would need re-pulling
with those columns — do not silently drop scenarios).

- [x] **Step 3: Verify the store reopens cold**

Run: `uv run --group fraud python -c "
import duckdb
con = duckdb.connect('projects/fraud_anomaly_detection/data/graph/fraud_graph.duckdb', read_only=True)
print(con.execute('SELECT key, value FROM meta').fetchall())"`
Expected: meta rows including built_at, source, n_edges and per-type counts.

- [x] **Step 4: Commit**

```bash
git add projects/fraud_anomaly_detection/analysis/graph_store_demo.py
git commit -m "fraud: graph store capability demo on the local sample

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: DuckPGQ probe (`analysis/graph_pgq_probe.py`)

Scoped experiment per spec: can the community extension answer the 3-hop
question in SQL against our tables? Graceful skip if install fails (needs
network). Nothing depends on the outcome.

**Files:**
- Create: `projects/fraud_anomaly_detection/analysis/graph_pgq_probe.py`

- [x] **Step 1: Write the probe**

```python
"""DuckPGQ probe — scoped experiment, nothing depends on it (spec: probe only).

Question under test: can SQL/PGQ MATCH answer "users within 3 hops of a
fraud user" on our store, and does it agree with queries.near_flagged?
Builds single-key vertex/edge tables (DuckPGQ wants simple keys), creates a
property graph, runs the hop query, compares user sets and wall time.

    uv run --group fraud python -m projects.fraud_anomaly_detection.analysis.graph_pgq_probe
"""

from __future__ import annotations

import time
from pathlib import Path

import duckdb

from projects.fraud_anomaly_detection.graph.load import DEFAULT_LAYERS, load_graph
from projects.fraud_anomaly_detection.graph.queries import near_flagged

STORE = Path("projects/fraud_anomaly_detection/data/graph/fraud_graph.duckdb")


def main() -> None:
    con = duckdb.connect(str(STORE))  # writable: probe creates pg_* tables
    try:
        try:
            con.execute("INSTALL duckpgq FROM community")
            con.execute("LOAD duckpgq")
        except Exception as exc:  # noqa: BLE001 — verdict, not control flow
            print(f"VERDICT: SKIPPED — extension unavailable: {exc}")
            return

        layer_list = ", ".join(f"'{l}'" for l in DEFAULT_LAYERS)
        con.execute("""
            CREATE OR REPLACE TABLE pg_users AS
            SELECT u.user_id AS id, coalesce(max(a.is_fraud), 0) AS is_fraud
            FROM users u LEFT JOIN advances a USING (user_id) GROUP BY 1;
        """)
        con.execute(f"""
            CREATE OR REPLACE TABLE pg_entities AS
            SELECT DISTINCT entity_type || ':' || entity_value AS id
            FROM edges WHERE entity_type IN ({layer_list});
        """)
        con.execute(f"""
            CREATE OR REPLACE TABLE pg_edges AS
            SELECT DISTINCT user_id AS src, entity_type || ':' || entity_value AS dst
            FROM edges WHERE entity_type IN ({layer_list});
        """)
        con.execute("""
            CREATE PROPERTY GRAPH fraud_pg
            VERTEX TABLES (
                pg_users PROPERTIES (id, is_fraud) LABEL account,
                pg_entities PROPERTIES (id) LABEL resource
            )
            EDGE TABLES (
                pg_edges SOURCE KEY (src) REFERENCES pg_users (id)
                         DESTINATION KEY (dst) REFERENCES pg_entities (id)
                         LABEL touches
            );
        """)

        t0 = time.perf_counter()
        rows = con.execute("""
            SELECT DISTINCT u.id FROM GRAPH_TABLE (fraud_pg
                MATCH (u:account)-[e:touches]-{1,6}(f:account)
                WHERE f.is_fraud = 1 AND u.is_fraud = 0
                COLUMNS (u.id)
            ) t(id)
        """).fetchall()
        pgq_users = {r[0] for r in rows}
        pgq_secs = time.perf_counter() - t0
        print(f"SQL/PGQ 3-user-hop neighbours of fraud: {len(pgq_users):,}"
              f" users in {pgq_secs:.2f}s")

        t0 = time.perf_counter()
        g = load_graph(STORE, scenarios=False, node_attrs=("is_fraud",))
        ig_users = set(near_flagged(g, flag="is_fraud", max_hops=3)["user_id"])
        ig_secs = time.perf_counter() - t0
        print(f"igraph near_flagged equivalent:        {len(ig_users):,}"
              f" users in {ig_secs:.2f}s (incl. load)")

        if pgq_users == ig_users:
            print(f"VERDICT: AGREES — identical user sets; pgq {pgq_secs:.2f}s"
                  f" vs igraph {ig_secs:.2f}s")
        else:
            only_pgq, only_ig = pgq_users - ig_users, ig_users - pgq_users
            print(f"VERDICT: DISAGREES — only-pgq {len(only_pgq)},"
                  f" only-igraph {len(only_ig)} (investigate before trusting pgq)")
    finally:
        con.close()


if __name__ == "__main__":
    main()
```

- [x] **Step 2: Run the probe**

Run: `uv run --group fraud python -m projects.fraud_anomaly_detection.analysis.graph_pgq_probe`
Expected: either `VERDICT: AGREES/DISAGREES ...` with timings, or
`VERDICT: SKIPPED — ...` (offline / extension failure). Record the verdict
line — Task 8 writes it into LEARNINGS. A DISAGREES or SKIPPED verdict is a
valid outcome, not a task failure; syntax errors from the extension count as
SKIPPED (note the error text).

- [x] **Step 3: Commit**

```bash
git add projects/fraud_anomaly_detection/analysis/graph_pgq_probe.py
git commit -m "fraud: DuckPGQ probe — scoped SQL/PGQ experiment vs igraph baseline

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Full verification + LEARNINGS entry

**Files:**
- Modify: `projects/fraud_anomaly_detection/LEARNINGS.md` (prepend new entry under the header)

- [x] **Step 1: Run the full project test suite**

Run: `uv run --group fraud pytest projects/fraud_anomaly_detection/tests/ -q`
Expected: all pass (scenario tests AND the three new graph test files).

- [x] **Step 2: Lint**

Run: `uv run ruff check projects/fraud_anomaly_detection/graph projects/fraud_anomaly_detection/analysis/graph_store_demo.py projects/fraud_anomaly_detection/analysis/graph_pgq_probe.py`
Expected: clean (fix anything it flags).

- [x] **Step 3: Re-run the demo end-to-end once more**

Run: `uv run --group fraud python -m projects.fraud_anomaly_detection.analysis.graph_store_demo`
Expected: completes; capture the printed numbers for the LEARNINGS entry.

- [x] **Step 4: Write the LEARNINGS entry**

Prepend under the file header of `projects/fraud_anomaly_detection/LEARNINGS.md`
(newest first — above the 2026-06-09 graph/entity-ring entry), filling the
bracketed numbers from the actual demo/probe output (the brackets are
execution-time measurements, not plan placeholders):

```markdown
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
the fraud-farm signal; the cap is only a traversal view choice).

**Capability demo on the 20k fraud-enriched sample** (graph_store_demo;
sample is graph-thinned — numbers are capability evidence, NOT transferable):
build [N] edges across [K] entity types in [T]s; scenario overlay flags
[N] users against the current register dynamically; [N] users within 3
user-hops of a flagged user; [N] multi-type (>=3 users, >=2 types)
components holding [N] fraud users; hub report top entity [value/type]
with [N] users.

**DuckPGQ probe verdict:** [AGREES/DISAGREES/SKIPPED — one line with timings
or reason]. [If AGREES: viable SQL front-end candidate at sample scale;
re-probe at v3 scale before relying on it. If SKIPPED/DISAGREES: igraph
remains the only traversal engine; revisit optional.]

**Next (unchanged from the v3 plan):** re-point the build at `v2_2ac98b52`
(full v3) — the store schema and views carry over as-is; the value question
(does multi-type density + new node types move coverage off ~0.01%?) is
TODO #2, now answerable with persistent infrastructure instead of one-off
scripts.
```

- [x] **Step 5: Commit**

```bash
git add projects/fraud_anomaly_detection/LEARNINGS.md
git commit -m "fraud: LEARNINGS — graph store capability demo + DuckPGQ verdict

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Verification checklist (whole plan)

- [x] `uv run --group fraud pytest projects/fraud_anomaly_detection/tests/ -q` — all green
- [x] `uv run --group fraud python -m projects.fraud_anomaly_detection.analysis.graph_store_demo` — 7 banners, store file present
- [x] `uv run --group fraud python -m projects.fraud_anomaly_detection.analysis.graph_pgq_probe` — a VERDICT line
- [x] `git status` — clean (no stray `.duckdb` tracked; `data/graph/` ignored)
- [x] Spec cross-check: lossless store (no cap in build.py), self-contained (advances snapshot queried by load), scenario overlay dynamic (register read at load time), hub report uncapped, projection capped by default
```
