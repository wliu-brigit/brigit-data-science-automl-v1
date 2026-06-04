# Step 1 — Keys & naming cleanup (no Snowflake)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `hash_key` → `unique_key` everywhere, add `split_group_key`
(defaults to `unique_key`), delete the row fallback, `SPLITID` → `SPLIT_PCT`
(+ helper renames), and add the materialize-edge validation (unique_key
present + no duplicates; SPLIT_PCT present/integer/0–99; loud collision
error when a file source already provides the split column).

**Architecture:** Two independent mechanical sweeps, landed as two green
commits: Part A renames the split column vocabulary (`SPLITID`→`SPLIT_PCT`,
`split_id_col`→`split_pct_col`, `add_split_id`→`add_split_pct`); Part B
replaces the key vocabulary (`hash_key`→`unique_key`+`split_group_key`,
fallback removal, new hard checks). Behavior-preserving for well-formed
projects; loud for latent duplicates. Forward-only — no back-compat keys in
`from_dict`, old logged state is disposable (wendao wipes manually).

**Tech stack:** Python 3.12, pandas, pytest. Everything through `uv run`.

**Source of truth:** `../design.md` §7 (keys), §8 (SPLIT_PCT), §11
(validation), §14 step 1. Open items resolved in conversation 2026-06-04:
file-source SPLIT_PCT collision → **error** (symmetry with Snowflake);
public key surface = source properties `unique_key_columns` /
`split_group_key_columns`, shared normalizer stays module-internal.

**Ground rules (all steps):** never delete/migrate MLflow runs, GCS objects,
or warehouse tables; `projects/fraud_anomaly_detection/` touched only where
the contract change requires (listed explicitly below); everything through
`uv`; credentials only via `.env`.

---

## Rename map (the whole step in one table)

| Old | New | Scope |
|---|---|---|
| `"SPLITID"` literal | `"SPLIT_PCT"` | column name everywhere |
| `split_id_col` (params, fields, dict keys) | `split_pct_col` | `DataPipeline`, `Dataset`, `DatasetRef`, `FeatureRegistry.build_from_df`, `EvalDataset`, identity payloads, JSON keys |
| `add_split_id()` | `add_split_pct()` | `automl/data/split.py` + exports |
| `hash_key` (params, fields, dict keys) | `unique_key` | sources, `Dataset`, eval layer, identity payloads, JSON keys |
| — (new) | `split_group_key` (optional, defaults to `unique_key`) | sources, `Dataset` |
| `HashKey` type alias | `Key` | `automl/data/split.py` + exports |
| `hash_key_columns()` (public) | `_normalize_key()` (internal) + source properties `unique_key_columns` / `split_group_key_columns` | `split.py`, `sources/base.py` |
| `ROW_FALLBACK_HASH_KEY`, `hash_key=None` branch | deleted | `split.py`, `pipeline.py`, the test pinning it |

Files touched (from grep, 2026-06-04; review-amended): `automl/data/{split,
pipeline,dataset,contract,features,registry,profile,spec}.py`,
`automl/data/sources/{base,local_csv,gcs_parquet,snowflake}.py`,
`automl/data/__init__.py`,
`automl/eval/{eval_dataset,prepare,_load,base,evaluate,results}.py`,
`automl/project/scaffold.py`, `projects/example_homecredit/config.py`,
`projects/example_homecredit/model/__init__.py`,
`projects/example_homecredit/PROJECT_INSTRUCTIONS.md`,
`projects/fraud_anomaly_detection/{config.py,data/queries/training_data.sql}`,
`projects/payment_routing/{config.py,data/queries/training_data.sql}`
(**git-tracked**; its `SnowflakeSource(...)` would TypeError once
`unique_key` is required — same minimal touch as fraud),
`agent-skills/references/setup/{data-pipeline,run-config}.md`, ~24 test files
(enumerated in Tasks A6/B9).

---

## PART A — `SPLITID` → `SPLIT_PCT` (commit 1)

### Task A1: split.py column rename

**Files:**
- Modify: `automl/data/split.py`
- Modify: `automl/data/__init__.py`

- [ ] **Step 1: Rename in `automl/data/split.py`**

Replace the module with (Part B rewrites it again — this pass is the
column/function rename only; `hash_key` machinery survives until B1):

```python
"""Deterministic split-bucket helpers for data materialization."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import pandas as pd


HashKey: TypeAlias = str | Sequence[str]
ROW_FALLBACK_HASH_KEY = "__row_fallback__"
SPLIT_PCT_COL = "SPLIT_PCT"


def hash_key_columns(hash_key: HashKey | None) -> tuple[str, ...]:
    """Normalize a hash-key declaration."""
    if hash_key is None:
        return (ROW_FALLBACK_HASH_KEY,)
    if isinstance(hash_key, str):
        columns = (hash_key,)
    else:
        try:
            columns = tuple(hash_key)
        except TypeError as exc:
            raise ValueError(
                "hash_key must be a column name, a non-empty sequence, or None"
            ) from exc
    if not columns or any(not isinstance(column, str) or not column.strip() for column in columns):
        raise ValueError("hash_key must contain non-empty column names")
    if len(set(columns)) != len(columns):
        raise ValueError("duplicate hash_key columns are not allowed")
    return tuple(sorted(columns))


def add_split_pct(
    df: pd.DataFrame,
    *,
    hash_key: HashKey | None,
    split_pct_col: str = SPLIT_PCT_COL,
    source_label: str = "data",
) -> pd.DataFrame:
    """Return ``df`` with deterministic 0..99 split buckets in ``split_pct_col``."""
    out = df.loc[:, [column for column in df.columns if column != split_pct_col]].copy()
    if hash_key is None:
        split_pct = pd.util.hash_pandas_object(out, index=True).mod(100).astype("int64")
        out[split_pct_col] = split_pct.to_numpy()
        return out

    columns = hash_key_columns(hash_key)
    missing = [column for column in columns if column not in out.columns]
    if missing:
        raise KeyError(
            f"hash_key column(s) {missing} not in {source_label} columns: {list(out.columns)}"
        )
    split_pct = (
        pd.util.hash_pandas_object(out[list(columns)], index=False)
        .mod(100)
        .astype("int64")
    )
    out[split_pct_col] = split_pct.to_numpy()
    return out


def split_report(
    df: pd.DataFrame,
    *,
    split_pct_col: str = SPLIT_PCT_COL,
) -> pd.DataFrame:
    if split_pct_col not in df.columns:
        raise KeyError(f"split_report requires {split_pct_col!r}")
    counts = df[split_pct_col].value_counts().sort_index()
    return pd.DataFrame({"bucket": counts.index.astype(int), "rows": counts.to_numpy()})


__all__ = [
    "HashKey",
    "ROW_FALLBACK_HASH_KEY",
    "SPLIT_PCT_COL",
    "add_split_pct",
    "hash_key_columns",
    "split_report",
]
```

- [ ] **Step 2: Update the export in `automl/data/__init__.py`**

```python
from automl.data.split import HashKey, add_split_pct, hash_key_columns, split_report
```

and in `__all__`: `"add_split_id"` → `"add_split_pct"`.

### Task A2: pipeline, dataset, contract, features, registry

**Files:**
- Modify: `automl/data/pipeline.py`
- Modify: `automl/data/dataset.py`
- Modify: `automl/data/contract.py`
- Modify: `automl/data/features.py`
- Modify: `automl/data/registry.py`

- [ ] **Step 1: `automl/data/pipeline.py`**

```python
# import line 16:
from automl.data.split import ROW_FALLBACK_HASH_KEY, add_split_pct, hash_key_columns
# class attr (line 29):
class DataPipeline:
    split_pct_col = "SPLIT_PCT"
# run() line 51:
        df = add_split_pct(df, hash_key=None if hash_key == (ROW_FALLBACK_HASH_KEY,) else hash_key)
# run() line 57 (registry build):
            split_pct_col=self.split_pct_col,
# _dataset_for(): identity payload key "split_id_col" → "split_pct_col",
#   value self.split_pct_col; Dataset(...) kwarg split_id_col= → split_pct_col=
# _validate_existing_dataset_matches_candidate field tuple: "split_id_col" → "split_pct_col"
```

- [ ] **Step 2: `automl/data/dataset.py`** — `Dataset.split_id_col` field →
`split_pct_col`; `from_dict`: `split_pct_col=str(payload.get("split_pct_col", "SPLIT_PCT"))`
(no `split_id_col` fallback key — forward-only); `to_dict` key
`"split_id_col"` → `"split_pct_col"`.

- [ ] **Step 3: `automl/data/contract.py`** — `DatasetRef.split_id_col` field →
`split_pct_col` (dataclass field, `from_dataset`, `from_dict` key, and the
`validate_trial_data_contract` checks-dict key).

- [ ] **Step 4: `automl/data/features.py`** — `build_from_df` parameter
`split_id_col: str = "SPLITID"` → `split_pct_col: str = "SPLIT_PCT"`; the
`is_metadata` comparison uses the renamed parameter.

- [ ] **Step 5: `automl/data/registry.py`** — line 80:
`df[df[dataset.split_pct_col].isin(buckets)]`.

### Task A3: eval layer column rename

**Files:**
- Modify: `automl/eval/eval_dataset.py`
- Modify: `automl/eval/prepare.py`

- [ ] **Step 1: `automl/eval/eval_dataset.py`** — rename in five places:
`compute_eval_dataset_identity` parameter `split_id_col` → `split_pct_col`
and its payload key `"split_id_col"` → `"split_pct_col"` (line 41 — this
changes `ev_` ids; forward-only, accepted); the `split_view`-kind required
check message; `EvalDataset.split_id_col` field → `split_pct_col`;
`EvalDataset.split_view(... split_id_col=...)` parameter → `split_pct_col`;
`from_dict`/`to_dict` keys.

- [ ] **Step 2: `automl/eval/prepare.py`** — `_prepare_split_view` line 119:
`split_pct_col=parent.split_pct_col`.

### Task A4: scaffold + projects column rename

**Files:**
- Modify: `automl/project/scaffold.py`
- Modify: `projects/example_homecredit/model/__init__.py`
- Modify: `projects/example_homecredit/PROJECT_INSTRUCTIONS.md`
- Modify: `projects/fraud_anomaly_detection/data/queries/training_data.sql`
- Modify: `projects/payment_routing/data/queries/training_data.sql`

- [ ] **Step 1:** In `scaffold.py` `_snowflake_templates()`, the
training_data.sql starter line becomes:

```sql
    MOD(ABS(HASH(<TBD_HASH_KEY_COLUMN>)), 100) AS SPLIT_PCT
```

(Step 3 of the effort rewrites this template entirely; this pass only keeps
the literal consistent. `<TBD_HASH_KEY_COLUMN>` is renamed in Part B.)

- [ ] **Step 2:** `projects/example_homecredit/model/__init__.py:43` — the
string literal in the `exclude={target, "SPLITID", *required_columns}` set
becomes `"SPLIT_PCT"`.

- [ ] **Step 3:** `projects/example_homecredit/PROJECT_INSTRUCTIONS.md:25` —
"`SPLITID` is derived by stable hash from `SK_ID_CURR`" → `SPLIT_PCT`.

- [ ] **Step 4:** `projects/fraud_anomaly_detection/data/queries/training_data.sql`
— `AS SPLITID` → `AS SPLIT_PCT` (explicitly required by the contract rename;
no other fraud changes in Part A).

- [ ] **Step 5:** `projects/payment_routing/data/queries/training_data.sql`
— lines 4 and 9: `SPLITID` → `SPLIT_PCT` (this file hashes `payment_id`
directly; just the literal renames, nothing else).

### Task A5: docs touched by the column rename

**Files:**
- Modify: `agent-skills/references/setup/data-pipeline.md`
- Modify: `agent-skills/references/setup/run-config.md`

- [ ] **Step 1:** `grep -n "SPLITID\|split_id_col\|add_split_id" agent-skills/references/setup/*.md`
and update every hit to the new names (`SPLIT_PCT`, `split_pct_col`,
`add_split_pct`). **Scope guard for `run-config.md:29`:** swap the `SPLITID`
literal only — the surrounding `Splits(train=[(start, end)])` range prose
stays as-is (the range API is correct until step 4 cuts it; don't pull
predicate language in early). Do not touch `docs/archive/**` (history stays
as written).

### Task A6: test sweep for the column rename

**Files (every test file with `SPLITID`/`split_id_col`/`add_split_id` hits):**
- `tests/unit/data/test_sources_pipeline_contract.py`
- `tests/unit/data/test_contract_validators.py`
- `tests/unit/data/test_materialize_return_shape.py`
- `tests/unit/data/test_profile.py`
- `tests/unit/eval/test_eval_dataset_identity.py`
- `tests/unit/eval/test_eval_thin_path.py`
- `tests/unit/agent/test_proposer_context.py`
- `tests/unit/cli/test_cli_catalog.py`
- `tests/integration/data_pipeline/test_materialize_load.py`
- `tests/integration/data_pipeline/test_trial_replay.py`
- `tests/integration/eval/test_eval_dataset_persistence.py`
- `tests/integration/homecredit/test_required_transformer_fixture.py`
- `tests/e2e/test_homecredit_data_model_breadth.py`

- [ ] **Step 1: Mechanical sweep.** For each file apply the rename map
(`"SPLITID"` → `"SPLIT_PCT"`, `split_id_col=` → `split_pct_col=`,
`add_split_id` → `add_split_pct`, `.split_id_col` → `.split_pct_col`).
These are fixture constructions and assertions pinning today's shape —
the *assertion values* change with the shape, which is the contract-tests
rule working as intended.

- [ ] **Step 2: Verify nothing is left**

```bash
grep -rn "SPLITID\|split_id_col\|add_split_id" automl tests projects agent-skills --include="*.py" --include="*.md" --include="*.sql" | grep -v docs/archive
```

Expected: no output.

### Task A7: green suite, commit 1

- [ ] **Step 1:** `uv run pytest tests/unit tests/contracts tests/integration`
— Expected: all pass.

- [ ] **Step 2: Commit**

```bash
git add -A automl tests projects agent-skills
git commit -m "Rename SPLITID -> SPLIT_PCT across the data/eval surface

split_id_col -> split_pct_col, add_split_id -> add_split_pct; contract
tests updated with the shapes they pin (design step 1, part A)."
```

---

## PART B — keys: `unique_key` + `split_group_key`, fallback removal, validation (commit 2)

### Task B1: split.py — keys, validation, collision (TDD)

**Files:**
- Create: `tests/unit/data/test_split_keys.py`
- Modify: `automl/data/split.py`
- Modify: `automl/data/__init__.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Key normalization, SPLIT_PCT assignment, and materialize-edge validation."""

import pandas as pd
import pytest

from automl.data.split import (
    SPLIT_PCT_COL,
    add_split_pct,
    split_report,
    validate_split_pct,
    validate_unique_key,
)
from automl.errors import DataError

pytestmark = pytest.mark.unit


def _frame():
    return pd.DataFrame({"user_id": ["a", "b", "c", "a"], "txn_id": [1, 2, 3, 4], "x": [0.1, 0.2, 0.3, 0.4]})


def test_add_split_pct_assigns_deterministic_buckets_from_group_key():
    out1 = add_split_pct(_frame(), split_group_key=("user_id",))
    out2 = add_split_pct(_frame(), split_group_key=("user_id",))
    assert SPLIT_PCT_COL in out1.columns
    assert out1[SPLIT_PCT_COL].between(0, 99).all()
    assert out1[SPLIT_PCT_COL].tolist() == out2[SPLIT_PCT_COL].tolist()
    # same group key value -> same bucket (rows 0 and 3 share user_id "a")
    assert out1[SPLIT_PCT_COL].iloc[0] == out1[SPLIT_PCT_COL].iloc[3]


def test_add_split_pct_errors_when_source_already_provides_the_column():
    df = _frame()
    df[SPLIT_PCT_COL] = 0
    with pytest.raises(DataError, match="SPLIT_PCT"):
        add_split_pct(df, split_group_key=("user_id",))


def test_add_split_pct_errors_on_missing_group_key_column():
    with pytest.raises(KeyError, match="split_group_key"):
        add_split_pct(_frame(), split_group_key=("nope",))


def test_validate_unique_key_passes_for_unique_tuples():
    validate_unique_key(_frame(), unique_key=("txn_id",))
    validate_unique_key(_frame(), unique_key=("txn_id", "user_id"))


def test_validate_unique_key_errors_on_duplicates_with_examples():
    with pytest.raises(DataError, match="duplicate"):
        validate_unique_key(_frame(), unique_key=("user_id",))


def test_validate_unique_key_errors_on_missing_columns():
    with pytest.raises(DataError, match="unique_key"):
        validate_unique_key(_frame(), unique_key=("nope",))


def test_validate_split_pct_accepts_integer_0_99():
    df = _frame()
    df[SPLIT_PCT_COL] = [0, 50, 99, 7]
    validate_split_pct(df)


@pytest.mark.parametrize(
    "values, match",
    [([0, 50, 99, 100], "0–99|0-99"), ([0.5, 1.0, 2.0, 3.0], "integer"), (None, "missing")],
)
def test_validate_split_pct_rejects_bad_columns(values, match):
    df = _frame()
    if values is not None:
        df[SPLIT_PCT_COL] = values
    with pytest.raises(DataError, match=match):
        validate_split_pct(df)


def test_split_report_counts_buckets():
    df = add_split_pct(_frame(), split_group_key=("txn_id",))
    report = split_report(df)
    assert int(report["rows"].sum()) == len(df)
```

- [ ] **Step 2: Run to verify failure**

`uv run pytest tests/unit/data/test_split_keys.py -v`
Expected: FAIL — `ImportError: cannot import name 'validate_split_pct'`.

- [ ] **Step 3: Rewrite `automl/data/split.py`**

```python
"""Deterministic split-bucket helpers, key normalization, and ingestion-edge checks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import pandas as pd

from automl.errors import DataError


Key: TypeAlias = str | Sequence[str]
SPLIT_PCT_COL = "SPLIT_PCT"


def _normalize_key(key: Key, *, field_name: str) -> tuple[str, ...]:
    """Normalize a key declaration to a sorted tuple of column names."""
    if isinstance(key, str):
        columns = (key,)
    else:
        try:
            columns = tuple(key)
        except TypeError as exc:
            raise ValueError(
                f"{field_name} must be a column name or a non-empty sequence of column names"
            ) from exc
    if not columns or any(not isinstance(column, str) or not column.strip() for column in columns):
        raise ValueError(f"{field_name} must contain non-empty column names")
    if len(set(columns)) != len(columns):
        raise ValueError(f"duplicate {field_name} columns are not allowed")
    return tuple(sorted(columns))


def add_split_pct(
    df: pd.DataFrame,
    *,
    split_group_key: tuple[str, ...],
    split_pct_col: str = SPLIT_PCT_COL,
    source_label: str = "data",
) -> pd.DataFrame:
    """Return ``df`` with deterministic 0..99 split buckets hashed from ``split_group_key``.

    The source must not already provide the column: a pre-existing split
    column has ambiguous provenance, so it is a loud error (symmetric with
    SnowflakeSource's injection collision error), never silently recomputed.
    """
    if split_pct_col in df.columns:
        raise DataError(
            f"{source_label} already provides a {split_pct_col} column; the pipeline computes "
            f"it from split_group_key — rename or remove the source column"
        )
    missing = [column for column in split_group_key if column not in df.columns]
    if missing:
        raise KeyError(
            f"split_group_key column(s) {missing} not in {source_label} columns: {list(df.columns)}"
        )
    split_pct = (
        pd.util.hash_pandas_object(df[list(split_group_key)], index=False)
        .mod(100)
        .astype("int64")
    )
    out = df.copy()
    out[split_pct_col] = split_pct.to_numpy()
    return out


def validate_unique_key(
    df: pd.DataFrame,
    *,
    unique_key: tuple[str, ...],
    source_label: str = "data",
) -> None:
    """Hard ingestion-edge check: unique_key columns present and duplicate-free."""
    missing = [column for column in unique_key if column not in df.columns]
    if missing:
        raise DataError(
            f"unique_key column(s) {missing} not in {source_label} columns: {list(df.columns)}"
        )
    duplicated = df.duplicated(subset=list(unique_key))
    if bool(duplicated.any()):
        examples = (
            df.loc[df.duplicated(subset=list(unique_key), keep=False), list(unique_key)]
            .head(5)
            .to_dict("records")
        )
        raise DataError(
            f"unique_key {unique_key} has {int(duplicated.sum())} duplicate row(s) in "
            f"{source_label}; examples: {examples}"
        )


def validate_split_pct(
    df: pd.DataFrame,
    *,
    split_pct_col: str = SPLIT_PCT_COL,
    source_label: str = "data",
) -> None:
    """Hard ingestion-edge check: split column present, integer, in 0–99."""
    if split_pct_col not in df.columns:
        raise DataError(
            f"{split_pct_col} missing from {source_label}; carry {split_pct_col} through "
            "from the base table"
        )
    series = df[split_pct_col]
    if not pd.api.types.is_integer_dtype(series):
        raise DataError(f"{split_pct_col} must be an integer column, got dtype {series.dtype}")
    if len(series) and not series.between(0, 99).all():
        raise DataError(f"{split_pct_col} values must be in 0–99")


def split_report(
    df: pd.DataFrame,
    *,
    split_pct_col: str = SPLIT_PCT_COL,
) -> pd.DataFrame:
    if split_pct_col not in df.columns:
        raise KeyError(f"split_report requires {split_pct_col!r}")
    counts = df[split_pct_col].value_counts().sort_index()
    return pd.DataFrame({"bucket": counts.index.astype(int), "rows": counts.to_numpy()})


__all__ = [
    "Key",
    "SPLIT_PCT_COL",
    "add_split_pct",
    "split_report",
    "validate_split_pct",
    "validate_unique_key",
]
```

(`HashKey`, `ROW_FALLBACK_HASH_KEY`, and `hash_key_columns` are gone;
`_normalize_key` is module-internal and consumed by `sources/base.py`.)

- [ ] **Step 4:** `automl/data/__init__.py` — update the import/`__all__`:

```python
from automl.data.split import Key, add_split_pct, split_report, validate_split_pct, validate_unique_key
```

`__all__`: remove `"HashKey"`, `"hash_key_columns"`; add `"Key"`,
`"validate_split_pct"`, `"validate_unique_key"`.

- [ ] **Step 5:** `uv run pytest tests/unit/data/test_split_keys.py -v` —
Expected: PASS (the rest of the suite is red until B2–B4; that is expected
mid-sweep).

### Task B2: sources — required `unique_key`, optional `split_group_key`

**Files:**
- Modify: `automl/data/sources/base.py`
- Modify: `automl/data/sources/local_csv.py`
- Modify: `automl/data/sources/gcs_parquet.py`
- Modify: `automl/data/sources/snowflake.py`
- Modify: `tests/unit/data/test_sources_breadth.py`

- [ ] **Step 1: `automl/data/sources/base.py`**

```python
"""Data source extension anchor."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from automl.data.split import Key, _normalize_key

if TYPE_CHECKING:
    from automl.data.pipeline import DataPipeline


class DataSource(ABC):
    kind = "base"
    unique_key: Key
    split_group_key: Key | None = None

    @property
    def unique_key_columns(self) -> tuple[str, ...]:
        """The declared stable row identifier, normalized."""
        return _normalize_key(self.unique_key, field_name="unique_key")

    @property
    def split_group_key_columns(self) -> tuple[str, ...]:
        """The key whose hash assigns split buckets; defaults to unique_key."""
        if self.split_group_key is None:
            return self.unique_key_columns
        return _normalize_key(self.split_group_key, field_name="split_group_key")

    @abstractmethod
    def load(self, *, project_dir: str | Path | None = None, nrows: int | None = None) -> pd.DataFrame:
        """Load raw rows into a DataFrame."""

    @abstractmethod
    def identity(self) -> dict[str, Any]:
        """Return deterministic source identity fields."""

    def artifact_files(self, pipeline: "DataPipeline") -> dict[str, Path]:
        """Return source trace artifacts to attach to project overview runs."""
        return {}


__all__ = ["DataSource"]
```

- [ ] **Step 2: `automl/data/sources/local_csv.py`**

```python
"""Local CSV data source."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from automl.data.sources.base import DataSource
from automl.data.split import Key


@dataclass(frozen=True)
class LocalCSVSource(DataSource):
    csv_path: str | Path
    unique_key: Key
    split_group_key: Key | None = None

    kind = "local_csv"

    def __post_init__(self) -> None:
        self.unique_key_columns  # validate declarations at construction
        self.split_group_key_columns

    def load(
        self,
        *,
        project_dir: str | Path | None = None,
        nrows: int | None = None,
    ) -> pd.DataFrame:
        path = Path(self.csv_path)
        csv_path = path if path.is_absolute() else Path(project_dir or Path.cwd()) / path
        return pd.read_csv(csv_path, nrows=nrows)

    def identity(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "csv_path": str(self.csv_path),
            "unique_key": list(self.unique_key_columns),
            "split_group_key": list(self.split_group_key_columns),
        }


__all__ = ["LocalCSVSource"]
```

- [ ] **Step 3: `automl/data/sources/gcs_parquet.py`** — same shape:
fields become `gcs_uri`, `unique_key: Key`, `split_group_key: Key | None = None`;
`__post_init__` keeps `gcs.parse_gcs_uri(self.gcs_uri)` and adds the two
property touches; `identity()` returns
`{"kind", "gcs_uri", "unique_key": list(...), "split_group_key": list(...)}`.

- [ ] **Step 4: `automl/data/sources/snowflake.py`** — the stub gains the key
fields (the source becomes real in step 3 of the effort):

```python
@dataclass(frozen=True)
class SnowflakeSource(DataSource):
    base_table: str
    base_data_sql: str | Path
    training_data_sql: str | Path
    unique_key: Key
    split_group_key: Key | None = None

    kind = "snowflake"
```

`identity()` replaces `"hash_key": []` with
`"unique_key": list(self.unique_key_columns)` and
`"split_group_key": list(self.split_group_key_columns)`. `load()` stub and
the env fields are unchanged. Add the `from automl.data.split import Key`
import.

- [ ] **Step 5: Update `tests/unit/data/test_sources_breadth.py`** — **all**
source constructions in the file gain `unique_key=` (use the column the
fixture already has) — that includes the GCS/CSV identity tests **and** the
`SnowflakeSource` load-stub test at ~line 54, which otherwise TypeErrors;
identity assertions swap `"hash_key": [...]` for
`"unique_key": [...]` + `"split_group_key": [...]` (same value when not
declared separately). Add one new test:

```python
def test_split_group_key_defaults_to_unique_key_and_overrides():
    default = LocalCSVSource(csv_path="x.csv", unique_key="TXN_ID")
    assert default.split_group_key_columns == ("TXN_ID",)
    grouped = LocalCSVSource(csv_path="x.csv", unique_key="TXN_ID", split_group_key="USER_ID")
    assert grouped.unique_key_columns == ("TXN_ID",)
    assert grouped.split_group_key_columns == ("USER_ID",)


def test_sources_require_unique_key():
    with pytest.raises(TypeError):
        LocalCSVSource(csv_path="x.csv")  # unique_key is required
```

- [ ] **Step 6:** `uv run pytest tests/unit/data/test_sources_breadth.py -v` —
Expected: PASS.

### Task B3: pipeline — normalized keys, validation, collision check

**Files:**
- Modify: `automl/data/pipeline.py`

- [ ] **Step 1: Imports and `run()`**

```python
from automl.data.split import add_split_pct, validate_split_pct, validate_unique_key
```

`run()` becomes:

```python
    def run(self) -> LoadedDataset:
        raw = self.spec.source.load(
            project_dir=self.session.config.project_dir,
            nrows=self.spec.dry_run_rows if self.session.dry_run else None,
        )
        df, original_names = self.standardize_columns(raw)
        self._check_split_pct_collision(original_names)
        unique_key = self._normalize_key_columns(self.spec.source.unique_key_columns, original_names)
        split_group_key = self._normalize_key_columns(
            self.spec.source.split_group_key_columns, original_names
        )
        target_column = self._normalized_target(original_names, df)
        metadata_cols = self._normalize_declared(self.spec.metadata_cols, original_names)
        registry_metadata_cols = _unique_tuple((*metadata_cols, *unique_key, *split_group_key))
        exclude_cols = self._normalize_declared(self.spec.exclude_cols, original_names)
        df = self._apply_quality_filters(
            df,
            protected_cols=(target_column, *unique_key, *split_group_key, *metadata_cols),
        )
        df = add_split_pct(
            df, split_group_key=split_group_key, split_pct_col=self.split_pct_col
        )
        validate_unique_key(df, unique_key=unique_key, source_label=self.spec.source.kind)
        validate_split_pct(df, split_pct_col=self.split_pct_col, source_label=self.spec.source.kind)
        registry = FeatureRegistry().build_from_df(
            df,
            target_column=target_column,
            metadata_cols=registry_metadata_cols,
            exclude_cols=exclude_cols,
            split_pct_col=self.split_pct_col,
            original_names=original_names,
        )
        dataset = self._dataset_for(
            df,
            registry,
            unique_key=unique_key,
            split_group_key=split_group_key,
            target_column=target_column,
        )
        return LoadedDataset(dataset=dataset, df=df, registry=registry)
```

- [ ] **Step 2: Replace `_normalized_hash_key` with the generic mapper + collision check**

```python
    def _normalize_key_columns(
        self, columns: tuple[str, ...], original_names: dict[str, str]
    ) -> tuple[str, ...]:
        raw_to_normalized = {raw: normalized for normalized, raw in original_names.items()}
        return tuple(
            raw_to_normalized.get(column, _normalize_column(column)) for column in columns
        )

    def _check_split_pct_collision(self, original_names: dict[str, str]) -> None:
        collisions = [
            raw
            for normalized, raw in original_names.items()
            if normalized == self.split_pct_col.lower()
        ]
        if collisions:
            raise DataError(
                f"source column(s) {collisions} collide with {self.split_pct_col}: the pipeline "
                "computes it from split_group_key — rename or remove the source column"
            )
```

- [ ] **Step 3: `_dataset_for`** — signature
`(self, df, registry, *, unique_key, split_group_key, target_column, dataset_id="unmaterialized")`;
`source_identity["hash_key"] = list(hash_key)` becomes:

```python
        source_identity["unique_key"] = list(unique_key)
        source_identity["split_group_key"] = list(split_group_key)
```

identity-hash payload: `"hash_key": list(hash_key)` →
`"unique_key": list(unique_key)` + `"split_group_key": list(split_group_key)`;
`Dataset(...)` gains `unique_key=unique_key, split_group_key=split_group_key`
(replacing `hash_key=`).
`_validate_existing_dataset_matches_candidate` field tuple: `"hash_key"` →
`"unique_key"`, append `"split_group_key"`.

### Task B4: dataset + contract field renames

**Files:**
- Modify: `automl/data/dataset.py`
- Modify: `automl/data/contract.py`

- [ ] **Step 1: `Dataset`** — replace `hash_key: tuple[str, ...]` with:

```python
    unique_key: tuple[str, ...]
    split_group_key: tuple[str, ...]
```

`from_dict`:

```python
            unique_key=tuple(str(item) for item in payload.get("unique_key", ())),
            split_group_key=tuple(str(item) for item in payload.get("split_group_key", ())),
```

`to_dict`: `"unique_key": list(self.unique_key)`,
`"split_group_key": list(self.split_group_key)` (drop `"hash_key"`).

- [ ] **Step 2: `contract.py`** — no `hash_key` fields exist on
`DatasetRef`/`TrialDataContract`; nothing further here beyond Task A2's
`split_pct_col`. Verify: `grep -n "hash_key" automl/data/contract.py` →
no output.

### Task B5: eval layer — `hash_key` → `unique_key`

**Files:**
- Modify: `automl/eval/eval_dataset.py`
- Modify: `automl/eval/prepare.py`
- Modify: `automl/eval/_load.py`
- Modify: `automl/eval/base.py`
- Modify: `automl/eval/evaluate.py`
- Modify: `automl/eval/results.py`

- [ ] **Step 1: `eval_dataset.py`** — rename throughout (field on
`EvalDataset` and `Augmentation`, `compute_eval_dataset_identity` parameter
and payload key `"hash_key"` → `"unique_key"` (changes `ev_` ids;
forward-only), `split_view`/`external`/`Augmentation.create` parameters,
`from_dict`/`to_dict` keys, `_normalize_hash_key` → `_normalize_unique_key`
with message `"unique_key must contain at least one column"`, the duplicate-
and missing-column validation messages).

- [ ] **Step 2: `prepare.py`** — `prepare_eval_dataset(... hash_key=...)`
parameter → `unique_key=`; `_prepare_external(... hash_key=...)` →
`unique_key=`; `parent.hash_key` → `parent.unique_key`;
`Augmentation.create(... hash_key=base.hash_key)` →
`unique_key=base.unique_key`; `_validate_augmentation_against_eval_frame`
local renames.

- [ ] **Step 3: `_load.py`** — `LoadedEvalDataset.hash_key` field →
`unique_key`; `recipe.hash_key` reads → `recipe.unique_key`; error messages
("duplicate hash_key rows" → "duplicate unique_key rows").

- [ ] **Step 4: `base.py`** — `_hash_key_columns` → `_unique_key_columns`;
`_with_augmentation_frames(..., hash_key=...)` parameter → `unique_key=`;
messages ("missing hash_key columns" → "missing unique_key columns",
"duplicate hash_key rows" → "duplicate unique_key rows").

- [ ] **Step 5: `evaluate.py`** — `spec.evaluate(..., hash_key=loaded.hash_key)`
→ `unique_key=loaded.unique_key` (and the `evaluate` signature it calls in
`base.py`).

- [ ] **Step 6: `results.py`** — `Predictions.hash_key` field → `unique_key`;
`manifest_dict()` key `"hash_key"` → `"unique_key"`.

- [ ] **Step 7: Verify the eval sweep is complete**

```bash
grep -rn "hash_key" automl/eval
```

Expected: no output.

### Task B6: profile — unique-key cardinality observation

**Files:**
- Modify: `automl/data/profile.py`
- Modify: `tests/unit/data/test_profile.py`

- [ ] **Step 1: Add the stats check** (design §11: cardinality is surfaced,
not just trusted) — new function next to `_basic_observations`, registered
in `_STATS_CHECKS`:

```python
def _unique_key_cardinality(loaded: LoadedDataset) -> list[dict[str, Any]]:
    key = list(loaded.dataset.unique_key)
    if not key or any(column not in loaded.df.columns for column in key):
        return []
    n_distinct = int(len(loaded.df.drop_duplicates(subset=key)))
    return [
        {
            "kind": "unique_key_cardinality",
            "text": (
                f"unique_key {key}: {n_distinct} distinct of {len(loaded.df)} rows "
                f"({'1:1' if n_distinct == len(loaded.df) else 'DUPLICATES PRESENT'})."
            ),
            "source": "profile_deterministic",
        }
    ]


_STATS_CHECKS: list[StatsCheck] = [_basic_observations, _unique_key_cardinality]
```

- [ ] **Step 2:** In `tests/unit/data/test_profile.py`, fixtures already
construct `Dataset(...)` — update to `unique_key=`/`split_group_key=` (Task
B9 sweep) and add an assertion that the observations payload contains a
`"unique_key_cardinality"` entry.

### Task B7: scaffold + project configs

**Files:**
- Modify: `automl/project/scaffold.py`
- Modify: `projects/example_homecredit/config.py`
- Modify: `projects/fraud_anomaly_detection/config.py`
- Modify: `projects/fraud_anomaly_detection/data/queries/training_data.sql`
- Modify: `tests/unit/project/test_metadata_and_scaffold.py`

- [ ] **Step 1: `scaffold.py` `_CONFIG_TEMPLATE`** — the source block becomes:

```python
source = SnowflakeSource(
    base_table="<TBD_base_table>",  # the snapshot table your base-data SQL builds
    base_data_sql="data/queries/base_data.sql",
    training_data_sql="data/queries/training_data.sql",
    unique_key="<TBD_unique_key>",  # stable row identifier; tuple for composite keys
    # split_group_key="USER_ID",    # declare only when splits must group by a coarser key
)
# source = LocalCSVSource(csv_path=PROJECT_DIR / "data" / "my_data.csv", unique_key="ROW_ID")
# source = GCSParquetSource(gcs_uri="gs://bucket/path/data.parquet", unique_key="ROW_ID")
```

and the `metadata_cols` comment "e.g. the hash key" → "e.g. the unique key".
In `_snowflake_templates()`, `<TBD_HASH_KEY_COLUMN>` →
`<TBD_SPLIT_GROUP_KEY_COLUMN>`.

- [ ] **Step 2: `tests/unit/project/test_metadata_and_scaffold.py`** —
`CONFIG_PLACEHOLDERS` tuple gains `"<TBD_unique_key>"`; the SQL-placeholder
expectations swap `<TBD_HASH_KEY_COLUMN>` for `<TBD_SPLIT_GROUP_KEY_COLUMN>`.

- [ ] **Step 3: `projects/example_homecredit/config.py`** —
`HASH_KEY = "SK_ID_CURR"` → `UNIQUE_KEY = "SK_ID_CURR"`;
`LocalCSVSource(csv_path=SAMPLE_CSV, hash_key=HASH_KEY)` →
`LocalCSVSource(csv_path=SAMPLE_CSV, unique_key=UNIQUE_KEY)`;
`metadata_cols=(HASH_KEY,)` → `metadata_cols=(UNIQUE_KEY,)`. Keep the
explanatory comments in the same voice (this file is the self-teaching
example).

- [ ] **Step 4: Verify SK_ID_CURR is duplicate-free** (first contact with the
new hard check — if this fails, stop and surface it, don't work around):

```bash
uv run python -c "
import importlib, pandas as pd
cfg = importlib.import_module('projects.example_homecredit.config')
df = pd.read_csv(cfg.SAMPLE_CSV)
print('rows:', len(df), 'duplicate SK_ID_CURR:', int(df['SK_ID_CURR'].duplicated().sum()))
"
```

Expected: `duplicate SK_ID_CURR: 0`.

- [ ] **Step 5: `projects/fraud_anomaly_detection/config.py`** — minimal
contract-required touch only: add `unique_key="<TBD_unique_key>"` to the
`SnowflakeSource(...)` block (after `training_data_sql`). The SQL placeholder
in `data/queries/training_data.sql`: `<TBD_HASH_KEY_COLUMN>` →
`<TBD_SPLIT_GROUP_KEY_COLUMN>`. Nothing else in the project changes.

- [ ] **Step 6: `projects/payment_routing/config.py`** (git-tracked; review
finding — its module-level `SnowflakeSource(...)` at ~lines 27–30 would
`TypeError` on import once `unique_key` is required): the same minimal
touch — add `unique_key="<TBD_unique_key>"` (or `"payment_id"` if the
config's own SQL already names it as the row key — read the file and use
the honest value). Nothing else in the project changes.

### Task B8: docs touched by the key rename

**Files:**
- Modify: `agent-skills/references/setup/data-pipeline.md`
- Modify: `agent-skills/references/setup/run-config.md`

- [ ] **Step 1:** `grep -n "hash_key" agent-skills/references/setup/*.md` and
rewrite each hit in terms of `unique_key`/`split_group_key` (the data-pipeline
reference describes the source contract — it should now state: `unique_key`
required, the stable row identifier, hard-validated unique at materialize;
`split_group_key` optional, defaults to `unique_key`, controls split
grouping). Leave `docs/archive/**` untouched.

### Task B9: test sweep for the key rename

**Files (every remaining test with `hash_key` hits):**
- `tests/unit/data/test_sources_pipeline_contract.py` (also: delete the
  row-fallback test, add validation tests — see below)
- `tests/unit/data/test_contract_validators.py`
- `tests/unit/data/test_materialize_return_shape.py`
- `tests/unit/data/test_profile.py`
- `tests/unit/eval/test_eval_dataset_identity.py`
- `tests/unit/eval/test_augmentations.py`
- `tests/unit/eval/test_eval_thin_path.py`
- `tests/unit/eval/test_metrics_breadth.py`
- `tests/unit/eval/test_results_schemas.py`
- `tests/unit/mlflow/test_eval_predictions_artifacts.py`
- `tests/unit/agent/test_proposer_context.py`
- `tests/unit/cli/test_cli_catalog.py`
- `tests/unit/project/test_project_validation.py`
- `tests/integration/data_pipeline/test_materialize_load.py`
- `tests/integration/data_pipeline/test_profile_integration.py`
- `tests/integration/data_pipeline/test_trial_replay.py`
- `tests/integration/eval/test_augmentation_integration.py`
- `tests/integration/eval/test_eval_dataset_persistence.py`
- `tests/integration/eval/test_evaluate_persistence.py`
- `tests/integration/homecredit/test_required_transformer_fixture.py`
- `tests/e2e/test_eval_dataset_breadth.py`
- `tests/e2e/test_homecredit_data_model_breadth.py`

- [ ] **Step 1: Mechanical sweep** per the rename map: `hash_key=` →
`unique_key=` in source/`Dataset`/eval constructors and keyword calls;
`.hash_key` attribute reads → `.unique_key`; `"hash_key"` dict keys →
`"unique_key"`; `Dataset(...)` fixtures additionally gain
`split_group_key=` (same value as `unique_key` unless the test exercises
grouping).

- [ ] **Step 2: Delete the row-fallback test** in
`tests/unit/data/test_sources_pipeline_contract.py`
(`test_build_dataset_without_hash_key_uses_deterministic_row_fallback`,
~line 311) — the path no longer exists.

- [ ] **Step 3: Add pipeline-edge validation tests** to
`tests/unit/data/test_sources_pipeline_contract.py` (these exercise the
checks through `build_dataset`, where unit B1 exercised the helpers):

```python
def test_build_dataset_errors_on_duplicate_unique_key(tmp_path):
    csv_path = tmp_path / "dups.csv"
    pd.DataFrame({"row_id": [1, 1, 2], "x": [0.1, 0.2, 0.3], "target": [0, 1, 0]}).to_csv(
        csv_path, index=False
    )
    session = _session_for(tmp_path, csv_path, unique_key="row_id")  # use the file's existing session helper
    with pytest.raises(DataError, match="duplicate"):
        build_dataset(session=session)


def test_build_dataset_errors_when_source_provides_split_pct(tmp_path):
    csv_path = tmp_path / "collide.csv"
    pd.DataFrame({"row_id": [1, 2], "SPLIT_PCT": [3, 4], "target": [0, 1]}).to_csv(
        csv_path, index=False
    )
    session = _session_for(tmp_path, csv_path, unique_key="row_id")
    with pytest.raises(DataError, match="SPLIT_PCT"):
        build_dataset(session=session)


def test_build_dataset_groups_splits_by_split_group_key(tmp_path):
    csv_path = tmp_path / "grouped.csv"
    pd.DataFrame(
        {"txn_id": [1, 2, 3, 4], "user_id": ["a", "a", "b", "b"], "target": [0, 1, 0, 1]}
    ).to_csv(csv_path, index=False)
    session = _session_for(
        tmp_path, csv_path, unique_key="txn_id", split_group_key="user_id"
    )
    loaded = build_dataset(session=session)
    buckets = loaded.df.groupby("user_id")["SPLIT_PCT"].nunique()
    assert (buckets == 1).all()  # one user never straddles buckets
```

(Adapt `_session_for` to however the module builds sessions today —
it constructs `DataSpec(source=LocalCSVSource(...))` + `Session(...)`
inline; thread `unique_key`/`split_group_key` through that existing helper.)

- [ ] **Step 4: Verify nothing is left**

```bash
grep -rn "hash_key\|HashKey\|ROW_FALLBACK" automl tests projects agent-skills --include="*.py" --include="*.md" | grep -v docs/archive
```

Expected: no output.

### Task B10: green suite, commit 2

- [ ] **Step 1:** `uv run pytest tests/unit tests/contracts tests/integration`
— Expected: all pass.

- [ ] **Step 2: Update `docs/HANDOFF.md`** — "Where things stand": step 1 of
the effort landed (keys & naming); next action becomes step 2 (dataset
record & lifecycle) and the manual MLflow/GCS state wipe before/with it.

- [ ] **Step 3: Commit**

```bash
git add -A automl tests projects agent-skills docs/HANDOFF.md
git commit -m "Replace hash_key with unique_key + split_group_key; add materialize-edge checks

unique_key is required on every source and hard-validated (present, no
duplicate tuples) at materialize; split_group_key defaults to unique_key
and drives SPLIT_PCT assignment; row fallback deleted; file sources that
already provide SPLIT_PCT now error loudly (design step 1, part B)."
```

---

## Self-review checklist (run before declaring step 1 done)

- [ ] `grep -rn "SPLITID\|split_id_col\|hash_key\|ROW_FALLBACK" automl tests projects agent-skills | grep -v docs/archive` → empty.
- [ ] Full suite green: `uv run pytest tests/unit tests/contracts tests/integration`.
- [ ] `uv run automl --project example_homecredit validate project` still passes structurally.
- [ ] Dataset identity intentionally changed (new payload keys) — no migration; confirm nothing tries to read old records.
- [ ] fraud project diff is exactly: `unique_key="<TBD_unique_key>"` line + two SQL literal renames.
