# Eval Snapshot and Prediction Versioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace today's in-MLflow `predictions.parquet` + free-form `reevaluate` with content-addressed eval snapshots, GCS-keyed predictions joinable by `hash_key`, additive augmentation manifests, and a single idempotent `evaluate` verb that upserts metrics per label on the trial run.

**Architecture:** Mirror the existing `automl/data/snapshot.py` pattern for a new `automl/eval/snapshot.py`. Eval snapshots are either `split_view` (pointer-only, derived from a data snapshot) or `external` (content-hashed labeled frame in GCS). Predictions live at `predictions/<eval_snapshot_id>/<trial_run_id>.parquet` with a JSON manifest. The trial run's `eval/` artifact tree is keyed by human-readable label (`train`, `test`, user-named) with an `eval/manifest.json` as the table of contents. `EvalSpec` keeps `primary=Auc()` as one slot and `metrics=[...]` for the others, but unifies storage so `primary` becomes a string pointer.

**Tech Stack:** Python 3.11+, `uv` for all package work (NEVER `pip`), `pytest` per the tiered test layout (`tests/unit`, `tests/integration`, `tests/contracts`, `tests/regression`), `mlflow` for the trial-run artifact store, GCS via `automl/io/gcs.py`, pandas for frame manipulation.

**Spec:** `docs/superpowers/specs/2026-05-17-eval-snapshot-and-prediction-versioning-design.md` (commits `baabf0d`, `2efb957`, `5e9158d`, `5ed2a03`).

**Hard cutover policy:** No backward compatibility. Old `eval/predictions.parquet` MLflow artifacts and the `automl.reevaluation` module are deleted, not migrated. Contract tests are updated in lockstep.

**Test tier rules (from `automl_dev/CLAUDE.md`):**
- `tests/unit/` — function/class-level, no live services, fast.
- `tests/integration/` — multi-module, may use file-backed MLflow + tmp dirs.
- `tests/contracts/` — ratchet tests pinning architectural invariants; **update in the same change that breaks the shape**.
- All tests run via `uv run pytest tests/<tier>/`.

---

## Pre-implementation audit corrections (2026-05-18)

These corrections come from auditing this plan against the final design doc and
the current code. They override task snippets below when there is a conflict.

1. **Keep one project-level `hash_key` concept, but use the current code's
   source declaration as the input.** Today `hash_key` lives on
   `LocalCSVSource` / `GCSParquetSource`, not on `DataSpec`. Do not add a new
   `DataSpec.hash_key` field. Instead, normalize `DATA.source.hash_key` once in
   `DataPipeline` after source column standardization, store it on the pipeline
   as `tuple[str, ...]`, and pass that normalized tuple to data snapshot
   identity, manifests, eval snapshots, predictions, and augmentations.

2. **Unify hash-key ordering before SPLITID derivation.** The final design says
   `hash_key` is a sorted list in identity payloads and manifests. To make
   "same field used for SPLITID derivation and per-row identity" literally
   true, update `automl.data.split.hash_key_columns()` / `add_split_id()` to use
   the same sorted normalization. Add tests proving `["CUST_ID", "AS_OF"]` and
   `["AS_OF", "CUST_ID"]` derive identical `SPLITID` values and identical
   snapshot identity.

3. **`prepare_eval_split_view` must inherit from the data snapshot.** The design
   explicitly says callers do not pass `hash_key` for split views. Implement
   `prepare_eval_split_view(ctx, data_snapshot_id, split_id_col, buckets, ...)`
   by loading the referenced data snapshot manifest, inheriting
   `target_column` and `hash_key`, and writing only the eval snapshot manifest.
   Do not keep the Task 10 signature that accepts `target_col` and `hash_key`.

4. **Evaluate idempotency must reuse cached prediction bytes.** The design says
   "predictions exist, report missing" and "new metric names" reuse predictions
   with no re-prediction. Add a predictions reader/existence helper for
   `predictions/<eval_snapshot_id>/<trial_run_id>.parquet` and compute new
   metrics from cached `y_pred`. Do not call `model.predict(...)` on the
   prediction-reuse path.

5. **Primary retargeting is a pointer update only.** Add an explicit task/test
   for re-running `evaluate(..., set_as_primary_label=True)` or an `EvalSpec`
   with the same metric set and a different primary. It must update
   `eval/manifest.json`, the per-label `report.json` primary string, and the
   unprefixed MLflow scalar metric without scoring, rewriting predictions, or
   recomputing metric values.

6. **Runner integration must account for model logging order.** Current
   `_execute.py` computes eval before `mlflow.pyfunc.log_model(...)`, while the
   new `evaluate(model_run_id=...)` flow loads the model from MLflow. Task 23
   must either move pyfunc logging before `evaluate(...)` or add a private
   in-memory model parameter used only by the trial-time runner. Keep the public
   API as `evaluate(model_run_id=..., eval_snapshot_id=...)`.

7. **Model-eval compatibility comes from the logged feature registry.** The
   design references `features/model_feature_registry.csv`; do not require the
   loaded pyfunc model to expose `required_input_columns` or
   `required_input_dtypes()`. Build compatibility from the logged model feature
   registry, compare required input columns and dtype strings against the eval
   snapshot frame, and fail with clear `ColumnMissing` / `DtypeMismatch` errors.

8. **Composite `hash_key` is always a tuple internally and a JSON list in
   artifacts.** Fix snippets/tests below that compare `hash_key` to a string,
   merge on a scalar key, or serialize a tuple directly into manifests. Pandas
   joins and duplicate checks must use `list(hash_key)`.

9. **Augmentation idempotency must run before overlap rejection for the same
   content.** Re-publishing the identical `(eval_snapshot_id, name,
   content_hash)` should be a no-op even though the augmentation columns overlap
   the existing `name__hash8` directory. Overlap rejection applies to other
   augmentation directories.

10. **Finish the placeholder tasks before dispatching workers.** Tasks 26, 27,
    and 32 contain `...` bodies. Replace them with concrete tests before
    implementation. Also add explicit cleanup updates for `automl/runner/_stages.py`,
    agent/skill/reference docs that mention `eval/predictions.parquet`, notebook
    5, fresh-shell `automl eval` verification, the Home Credit reset smoke, and
    the data-science usability audit required by `docs/goal1.md`.

## Phase 1 — Foundation: unified `hash_key` invariant

The whole spec leans on a single unified `hash_key` field — declared by
every project, validated for uniqueness at pipeline init, used for both
SPLITID derivation and per-row identity. **No separate `row_id_col`
concept; no auto-promotion.** This phase normalizes `hash_key` to a
canonical tuple-of-strings representation and threads it through the data
snapshot identity and manifest.

**Canonical internal form:** `hash_key` is always stored as a sorted
`tuple[str, ...]`. User-facing input accepts a single `str` or a `list[str]`
and normalizes via:

```python
def normalize_hash_key(hash_key: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(hash_key, str):
        return (hash_key,)
    return tuple(sorted(hash_key))
```

Every manifest field stores `hash_key` as a JSON list (the tuple
serialized). Every code path that needs to index a frame uses
`list(hash_key)` so pandas returns the DataFrame slice (works for single
or composite). Every uniqueness check uses
`df[list(hash_key)].duplicated().any()` (works on a multi-col DataFrame
too).

### Task 1: Add `hash_key` to `SnapshotIdentity` and the data snapshot manifest

**Files:**
- Modify: `automl/data/snapshot.py`
- Test: `tests/unit/test_data_snapshot.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_data_snapshot.py`:

```python
def test_snapshot_identity_includes_hash_key() -> None:
    mod = _load_module()
    identity_a = mod.compute_snapshot_identity(
        _df_data(),
        _registry(),
        target_column="TARGET",
        split_id_col="SPLITID",
        hash_key="A",
        source_identity=_source_identity(),
    )
    identity_b = mod.compute_snapshot_identity(
        _df_data(),
        _registry(),
        target_column="TARGET",
        split_id_col="SPLITID",
        hash_key=["A", "TARGET"],   # composite → different identity
        source_identity=_source_identity(),
    )
    assert identity_a.snapshot_identity_hash != identity_b.snapshot_identity_hash
    assert identity_a.hash_key == ("A",)
    assert identity_b.hash_key == ("A", "TARGET")


def test_snapshot_identity_rejects_non_unique_hash_key() -> None:
    mod = _load_module()
    df = pd.DataFrame({
        "A": [1, 1, 2, 3],     # duplicates
        "TARGET": [0, 1, 0, 1],
        "SPLITID": [5, 65, 75, 95],
    })
    with pytest.raises(ValueError, match="unique"):
        mod.compute_snapshot_identity(
            df,
            _registry(),
            target_column="TARGET",
            split_id_col="SPLITID",
            hash_key="A",
            source_identity=_source_identity(),
        )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_data_snapshot.py -v
```

Expected: FAIL — `compute_snapshot_identity` doesn't accept `hash_key`.

- [ ] **Step 3: Add `hash_key` (as a tuple) to `SnapshotIdentity` and `compute_snapshot_identity`**

In `automl/data/snapshot.py`:

```python
def normalize_hash_key(hash_key) -> tuple[str, ...]:
    """Normalize a user-supplied hash_key to a sorted tuple of column names."""
    if isinstance(hash_key, str):
        return (hash_key,)
    cols = list(hash_key)
    if not cols or any(not isinstance(c, str) or not c for c in cols):
        raise ValueError("hash_key must be a non-empty column name or list of names")
    return tuple(sorted(cols))


@dataclass(frozen=True)
class SnapshotIdentity:
    data_content_hash: str
    feature_registry_hash: str
    schema_hash: str
    source_identity_hash: str
    snapshot_identity_hash: str
    snapshot_hash8: str
    target_column: str
    split_id_col: str
    hash_key: tuple[str, ...]              # NEW


def compute_snapshot_identity(
    df_data: pd.DataFrame,
    registry_df: pd.DataFrame,
    *,
    target_column: str,
    split_id_col: str,
    hash_key,                              # str | list[str] | tuple[str, ...]
    source_identity: dict[str, Any],
) -> SnapshotIdentity:
    hash_key_tuple = normalize_hash_key(hash_key)
    missing = [c for c in hash_key_tuple if c not in df_data.columns]
    if missing:
        raise ValueError(f"hash_key column(s) {missing} not in data columns")
    if df_data[list(hash_key_tuple)].duplicated().any():
        raise ValueError(
            f"hash_key {list(hash_key_tuple)!r} is not unique per row; "
            "pick a finer key or declare a composite"
        )
    frozen_source_identity = _freeze_jsonable(source_identity)
    source_identity_payload = _thaw_jsonable(frozen_source_identity)
    data_hash = dataframe_content_hash(df_data)
    registry_hash = registry_content_hash(registry_df)
    schema = schema_hash(df_data)
    source_identity_hash = _json_hash(source_identity_payload)
    identity_payload = {
        "data_content_hash": data_hash,
        "feature_registry_hash": registry_hash,
        "schema_hash": schema,
        "source_identity_hash": source_identity_hash,
        "target_column": target_column,
        "split_id_col": split_id_col,
        "hash_key": list(hash_key_tuple),   # JSON-friendly list
    }
    identity_hash = _json_hash(identity_payload)
    return SnapshotIdentity(
        data_content_hash=data_hash,
        feature_registry_hash=registry_hash,
        schema_hash=schema,
        source_identity_hash=source_identity_hash,
        snapshot_identity_hash=identity_hash,
        snapshot_hash8=identity_hash.removeprefix("sha256:")[:8],
        target_column=target_column,
        split_id_col=split_id_col,
        hash_key=hash_key_tuple,
    )
```

Also update `_identity_payload(identity)` to include
`"hash_key": list(identity.hash_key)`; `build_data_manifest` adds the same
top-level field to the returned dict; `validate_data_manifest_v2`
re-derives identity using `manifest["hash_key"]`.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/unit/test_data_snapshot.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add automl/data/snapshot.py tests/unit/test_data_snapshot.py
git commit -m "Add unified hash_key (tuple) to data snapshot identity and manifest"
```

---

### Task 2: Validate `hash_key` uniqueness in `DataPipeline.prepare_data`

**Files:**
- Modify: `automl/data/pipeline.py`
- Test: `tests/unit/data/test_data_pipeline_hash_key.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/data/test_data_pipeline_hash_key.py`:

```python
from __future__ import annotations
import pandas as pd
import pytest

from automl.data.pipeline import DataPipeline


def test_validate_hash_key_uniqueness_accepts_single_column():
    df = pd.DataFrame({"LOAN_ID": [1, 2, 3], "TARGET": [0, 1, 0]})
    DataPipeline._validate_hash_key(hash_key="LOAN_ID", df=df)


def test_validate_hash_key_uniqueness_accepts_composite():
    df = pd.DataFrame({
        "CUST_ID": [1, 1, 2, 2],
        "AS_OF": ["2026-Q1", "2026-Q2", "2026-Q1", "2026-Q2"],
        "TARGET": [0, 1, 0, 1],
    })
    DataPipeline._validate_hash_key(hash_key=["CUST_ID", "AS_OF"], df=df)


def test_validate_hash_key_rejects_non_unique_single_column():
    df = pd.DataFrame({"LOAN_ID": [1, 1, 2], "TARGET": [0, 1, 0]})
    with pytest.raises(ValueError, match="unique"):
        DataPipeline._validate_hash_key(hash_key="LOAN_ID", df=df)


def test_validate_hash_key_rejects_non_unique_composite():
    df = pd.DataFrame({
        "CUST_ID": [1, 1, 2],
        "AS_OF":   ["Q1", "Q1", "Q2"],   # (1, Q1) appears twice
        "TARGET":  [0, 1, 0],
    })
    with pytest.raises(ValueError, match="unique"):
        DataPipeline._validate_hash_key(hash_key=["CUST_ID", "AS_OF"], df=df)


def test_validate_hash_key_rejects_missing_column():
    df = pd.DataFrame({"X": [1, 2]})
    with pytest.raises(ValueError, match="not in"):
        DataPipeline._validate_hash_key(hash_key="LOAN_ID", df=df)


def test_validate_hash_key_rejects_none():
    df = pd.DataFrame({"X": [1, 2]})
    with pytest.raises(ValueError, match="required"):
        DataPipeline._validate_hash_key(hash_key=None, df=df)
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/data/test_data_pipeline_hash_key.py -v
```

Expected: FAIL — `_validate_hash_key` doesn't exist.

- [ ] **Step 3: Implement the validator**

In `automl/data/pipeline.py`, add as a `@staticmethod` on `DataPipeline`:

```python
@staticmethod
def _validate_hash_key(
    *,
    hash_key,            # str | list[str] | tuple[str, ...] | None
    df: pd.DataFrame,
) -> tuple[str, ...]:
    """Validate hash_key columns exist in df and that (hash_key tuples) are unique.

    Returns the normalized hash_key as a sorted tuple of column names.
    Raises ValueError on any contract violation. No auto-derivation — the
    project must declare DATA.hash_key.
    """
    if hash_key is None:
        raise ValueError(
            "hash_key is required: declare DATA.hash_key in project.py "
            "(single column name or list of names; must be unique per row)"
        )
    cols = [hash_key] if isinstance(hash_key, str) else list(hash_key)
    if not cols or any(not isinstance(c, str) or not c for c in cols):
        raise ValueError(
            "hash_key must be a non-empty column name or list of column names"
        )
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"hash_key column(s) {missing} not in data columns: {list(df.columns)}"
        )
    if df[cols].duplicated().any():
        n_dup = int(df[cols].duplicated().sum())
        raise ValueError(
            f"hash_key {cols!r} is not unique per row "
            f"({n_dup} duplicate tuples); pick a finer key or declare a composite"
        )
    return tuple(sorted(cols))
```

Wire it into `prepare_data()`: after the materialized `df_data` is ready,
call `hash_key_tuple = self._validate_hash_key(hash_key=self.hash_key, df=df_data)`
and pass `hash_key=hash_key_tuple` through to `compute_snapshot_identity(...)`
and `build_data_manifest(...)`.

Add `hash_key` to `DataPipeline.__init__`. Source the value from
`DATA.hash_key` in `build_pipeline(ctx, ...)`.

- [ ] **Step 4: Run the test**

```bash
uv run pytest tests/unit/data/test_data_pipeline_hash_key.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add automl/data/pipeline.py tests/unit/data/test_data_pipeline_hash_key.py
git commit -m "Validate hash_key per-row uniqueness at pipeline init"
```

---

### Task 3: (removed — folded into Task 2)

Originally added an optional `DataSpec.row_id_col` field. Under the
unified contract there is no separate `row_id_col` concept; `DataSpec.hash_key`
already exists (single column or list) and is the sole project-level
declaration. Task numbers from here onward stay as written — do not
renumber, just skip this slot.

---

## Phase 2 — Eval snapshot identity and manifest

Build the symmetric `automl/eval/snapshot.py` module: identity hashing, manifest builders, manifest validators, GCS path helpers. This phase produces no user-facing API — it's the foundation Phase 3 builds on.

### Task 4: `EvalSnapshotIdentity` for `external` kind

**Files:**
- Create: `automl/eval/snapshot.py`
- Test: `tests/unit/test_eval_snapshot.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_eval_snapshot.py`:

```python
from __future__ import annotations

import pandas as pd
import pytest


def _df_external() -> pd.DataFrame:
    return pd.DataFrame({
        "SK_ID_CURR": [101, 102, 103, 104],
        "FEATURE_A": [1.0, 2.0, 3.0, 4.0],
        "TARGET": [0, 1, 0, 1],
    })


def test_external_eval_snapshot_identity_is_deterministic():
    from automl.eval.snapshot import compute_eval_snapshot_identity

    a = compute_eval_snapshot_identity(
        kind="external",
        df=_df_external(),
        target_column="TARGET",
        hash_key="SK_ID_CURR",
    )
    b = compute_eval_snapshot_identity(
        kind="external",
        df=_df_external(),
        target_column="TARGET",
        hash_key=["SK_ID_CURR"],   # list form must produce same identity
    )
    assert a.eval_snapshot_hash == b.eval_snapshot_hash
    assert a.snapshot_name == b.snapshot_name
    assert a.hash_key == ("SK_ID_CURR",)


def test_external_eval_snapshot_identity_changes_when_a_value_changes():
    from automl.eval.snapshot import compute_eval_snapshot_identity

    df_a = _df_external()
    df_b = _df_external()
    df_b.loc[0, "FEATURE_A"] = 999.0

    a = compute_eval_snapshot_identity(
        kind="external", df=df_a,
        target_column="TARGET", hash_key="SK_ID_CURR",
    )
    b = compute_eval_snapshot_identity(
        kind="external", df=df_b,
        target_column="TARGET", hash_key="SK_ID_CURR",
    )
    assert a.eval_snapshot_hash != b.eval_snapshot_hash


def test_external_eval_snapshot_identity_rejects_missing_hash_key():
    from automl.eval.snapshot import compute_eval_snapshot_identity

    df = _df_external().drop(columns=["SK_ID_CURR"])
    with pytest.raises(ValueError, match="hash_key"):
        compute_eval_snapshot_identity(
            kind="external", df=df,
            target_column="TARGET", hash_key="SK_ID_CURR",
        )


def test_external_eval_snapshot_identity_rejects_non_unique_hash_key():
    from automl.eval.snapshot import compute_eval_snapshot_identity

    df = pd.DataFrame({
        "SK_ID_CURR": [1, 1, 2],
        "TARGET": [0, 1, 0],
    })
    with pytest.raises(ValueError, match="unique"):
        compute_eval_snapshot_identity(
            kind="external", df=df,
            target_column="TARGET", hash_key="SK_ID_CURR",
        )


def test_external_eval_snapshot_identity_supports_composite_hash_key():
    from automl.eval.snapshot import compute_eval_snapshot_identity

    df = pd.DataFrame({
        "CUST_ID": [1, 1, 2, 2],
        "AS_OF":   ["Q1", "Q2", "Q1", "Q2"],
        "F":       [1.0, 2.0, 3.0, 4.0],
        "TARGET":  [0, 1, 0, 1],
    })
    a = compute_eval_snapshot_identity(
        kind="external", df=df, target_column="TARGET",
        hash_key=["CUST_ID", "AS_OF"],
    )
    assert a.hash_key == ("AS_OF", "CUST_ID")  # sorted
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_eval_snapshot.py -v
```

Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement `compute_eval_snapshot_identity` for external kind**

Create `automl/eval/snapshot.py`:

```python
"""Eval snapshot identity, manifests, and GCS path helpers."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

EvalSnapshotKind = Literal["split_view", "external"]

EVAL_SNAPSHOT_NAME_RE = re.compile(r"^v([1-9][0-9]*)_([0-9a-f]{8})$")
EVAL_SNAPSHOT_HASH8_RE = re.compile(r"^[0-9a-f]{8}$")


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_hash(payload: Any) -> str:
    return _sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str))


def _frame_content_hash(df: pd.DataFrame) -> str:
    payload = {
        "columns": list(df.columns),
        "dtypes": [str(dtype) for dtype in df.dtypes],
        "row_hashes": pd.util.hash_pandas_object(df, index=False).astype("uint64").tolist(),
    }
    return _json_hash(payload)


def _schema_hash(df: pd.DataFrame) -> str:
    payload = {
        "columns": list(df.columns),
        "dtypes": [str(dtype) for dtype in df.dtypes],
    }
    return _json_hash(payload)


def _normalize_hash_key(hash_key) -> tuple[str, ...]:
    """Normalize a user-supplied hash_key to a sorted tuple of column names."""
    if isinstance(hash_key, str):
        return (hash_key,)
    cols = list(hash_key)
    if not cols or any(not isinstance(c, str) or not c for c in cols):
        raise ValueError("hash_key must be a non-empty column name or list of names")
    return tuple(sorted(cols))


@dataclass(frozen=True)
class EvalSnapshotIdentity:
    kind: EvalSnapshotKind
    target_column: str
    hash_key: tuple[str, ...]                  # always a tuple
    schema_hash: str
    content_hash: str
    eval_snapshot_hash: str
    snapshot_hash8: str
    snapshot_name: str
    # split_view only:
    of_data_snapshot_id: str | None = None
    split_id_col: str | None = None
    buckets: tuple[tuple[int, int], ...] | None = None


def compute_eval_snapshot_identity(
    *,
    kind: EvalSnapshotKind,
    df: pd.DataFrame | None = None,
    target_column: str,
    hash_key,                                  # str | list[str] | tuple[str, ...]
    of_data_snapshot_id: str | None = None,
    split_id_col: str | None = None,
    buckets: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
    version: int = 1,
) -> EvalSnapshotIdentity:
    hash_key_tuple = _normalize_hash_key(hash_key)
    if kind == "external":
        if df is None:
            raise ValueError("df is required for kind='external'")
        missing = [c for c in hash_key_tuple if c not in df.columns]
        if missing:
            raise ValueError(
                f"hash_key column(s) {missing} not in df columns: {list(df.columns)}"
            )
        if df[list(hash_key_tuple)].duplicated().any():
            raise ValueError(
                f"hash_key {list(hash_key_tuple)!r} values must be unique per row"
            )
        if target_column not in df.columns:
            raise ValueError(f"target_column {target_column!r} not in df columns")
        schema = _schema_hash(df)
        content = _frame_content_hash(df)
        identity_payload = {
            "kind": "external",
            "target_column": target_column,
            "hash_key": list(hash_key_tuple),
            "schema_hash": schema,
            "content_hash": content,
        }
        identity_hash = _json_hash(identity_payload)
        hash8 = identity_hash.removeprefix("sha256:")[:8]
        return EvalSnapshotIdentity(
            kind="external",
            target_column=target_column,
            hash_key=hash_key_tuple,
            schema_hash=schema,
            content_hash=content,
            eval_snapshot_hash=identity_hash,
            snapshot_hash8=hash8,
            snapshot_name=f"v{version}_{hash8}",
        )
    raise ValueError(f"unknown kind={kind!r}")  # split_view in next task
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/unit/test_eval_snapshot.py -v
```

Expected: all four tests pass.

- [ ] **Step 5: Commit**

```bash
git add automl/eval/snapshot.py tests/unit/test_eval_snapshot.py
git commit -m "Eval snapshot identity for kind=external"
```

---

### Task 5: `EvalSnapshotIdentity` for `split_view` kind

**Files:**
- Modify: `automl/eval/snapshot.py`
- Modify: `tests/unit/test_eval_snapshot.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_eval_snapshot.py`:

```python
def test_split_view_eval_snapshot_identity_is_deterministic_from_inputs():
    from automl.eval.snapshot import compute_eval_snapshot_identity

    a = compute_eval_snapshot_identity(
        kind="split_view",
        target_column="TARGET",
        hash_key="SK_ID_CURR",
        of_data_snapshot_id="sha256:abcdef0123",
        split_id_col="SPLITID",
        buckets=[(80, 100)],
    )
    b = compute_eval_snapshot_identity(
        kind="split_view",
        target_column="TARGET",
        hash_key="SK_ID_CURR",
        of_data_snapshot_id="sha256:abcdef0123",
        split_id_col="SPLITID",
        buckets=[(80, 100)],
    )
    assert a.eval_snapshot_hash == b.eval_snapshot_hash
    assert a.kind == "split_view"
    assert a.of_data_snapshot_id == "sha256:abcdef0123"
    assert a.buckets == ((80, 100),)


def test_split_view_identity_changes_with_buckets():
    from automl.eval.snapshot import compute_eval_snapshot_identity

    a = compute_eval_snapshot_identity(
        kind="split_view", target_column="TARGET", hash_key="SK_ID_CURR",
        of_data_snapshot_id="sha256:abc", split_id_col="SPLITID",
        buckets=[(80, 100)],
    )
    b = compute_eval_snapshot_identity(
        kind="split_view", target_column="TARGET", hash_key="SK_ID_CURR",
        of_data_snapshot_id="sha256:abc", split_id_col="SPLITID",
        buckets=[(0, 80)],
    )
    assert a.eval_snapshot_hash != b.eval_snapshot_hash


def test_split_view_buckets_are_sort_normalized():
    from automl.eval.snapshot import compute_eval_snapshot_identity

    a = compute_eval_snapshot_identity(
        kind="split_view", target_column="TARGET", hash_key="SK_ID_CURR",
        of_data_snapshot_id="sha256:abc", split_id_col="SPLITID",
        buckets=[(80, 100), (0, 20)],
    )
    b = compute_eval_snapshot_identity(
        kind="split_view", target_column="TARGET", hash_key="SK_ID_CURR",
        of_data_snapshot_id="sha256:abc", split_id_col="SPLITID",
        buckets=[(0, 20), (80, 100)],
    )
    assert a.eval_snapshot_hash == b.eval_snapshot_hash
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_eval_snapshot.py -v
```

Expected: the three new tests FAIL — `split_view` branch raises `unknown kind`.

- [ ] **Step 3: Implement the `split_view` branch**

Replace the `raise ValueError(...)` line in `compute_eval_snapshot_identity` with:

```python
    if kind == "split_view":
        if not of_data_snapshot_id:
            raise ValueError("of_data_snapshot_id is required for kind='split_view'")
        if not split_id_col:
            raise ValueError("split_id_col is required for kind='split_view'")
        if not buckets:
            raise ValueError("buckets is required for kind='split_view'")
        normalized_buckets = tuple(sorted((int(lo), int(hi)) for lo, hi in buckets))
        identity_payload = {
            "kind": "split_view",
            "target_column": target_column,
            "hash_key": list(hash_key_tuple),
            "of_data_snapshot_id": of_data_snapshot_id,
            "split_id_col": split_id_col,
            "buckets": [list(b) for b in normalized_buckets],
        }
        identity_hash = _json_hash(identity_payload)
        hash8 = identity_hash.removeprefix("sha256:")[:8]
        return EvalSnapshotIdentity(
            kind="split_view",
            target_column=target_column,
            hash_key=hash_key_tuple,
            schema_hash="",
            content_hash="",
            eval_snapshot_hash=identity_hash,
            snapshot_hash8=hash8,
            snapshot_name=f"v{version}_{hash8}",
            of_data_snapshot_id=of_data_snapshot_id,
            split_id_col=split_id_col,
            buckets=normalized_buckets,
        )
    raise ValueError(f"unknown kind={kind!r}")
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_eval_snapshot.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add automl/eval/snapshot.py tests/unit/test_eval_snapshot.py
git commit -m "Eval snapshot identity for kind=split_view"
```

---

### Task 6: Eval snapshot GCS path helpers

**Files:**
- Modify: `automl/eval/snapshot.py`
- Test: `tests/unit/test_eval_snapshot.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_eval_snapshot.py`:

```python
def test_eval_snapshot_gcs_paths():
    from automl.eval.snapshot import eval_snapshot_gcs_paths

    paths = eval_snapshot_gcs_paths(
        bucket="my-bucket",
        gcs_prefix="automl",
        project_name="example_homecredit",
        experiment_id="example-homecredit",
        snapshot_name="v1_e8a4c102",
        dry_run=False,
    )
    assert paths["bucket"] == "my-bucket"
    assert paths["base_path"].endswith("/eval/snapshots/v1_e8a4c102")
    assert paths["manifest_path"].endswith("/manifest.json")
    assert paths["data_path"].endswith("/data.parquet")
    assert paths["manifest_uri"].startswith("gs://my-bucket/")


def test_eval_snapshot_gcs_paths_dry_run_routes_under_dry_run():
    from automl.eval.snapshot import eval_snapshot_gcs_paths

    paths = eval_snapshot_gcs_paths(
        bucket="b", gcs_prefix="automl",
        project_name="p", experiment_id="e",
        snapshot_name="v1_aaaaaaaa", dry_run=True,
    )
    assert "/dry_run/p/e/" in paths["manifest_path"]


def test_eval_augmentation_gcs_paths():
    from automl.eval.snapshot import eval_augmentation_gcs_paths

    paths = eval_augmentation_gcs_paths(
        bucket="b", gcs_prefix="automl",
        project_name="p", experiment_id="e",
        snapshot_name="v1_e8a4c102",
        augmentation_dir="ltv__a3f1c204",
    )
    assert paths["base_path"].endswith(
        "/eval/snapshots/v1_e8a4c102/augmentations/ltv__a3f1c204"
    )
    assert paths["manifest_uri"].endswith("/manifest.json")
    assert paths["data_uri"].endswith("/data.parquet")
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_eval_snapshot.py -v
```

Expected: 3 new FAIL — functions don't exist.

- [ ] **Step 3: Implement the helpers**

Add to `automl/eval/snapshot.py`:

```python
from automl.mlflow.artifacts.gcs_paths import route_prefix_for


def validate_eval_snapshot_name(value: str) -> None:
    if not EVAL_SNAPSHOT_NAME_RE.fullmatch(value):
        raise ValueError(
            "eval_snapshot_name must match v<version>_<hash8> with 8 lowercase hex characters"
        )


def eval_snapshot_gcs_paths(
    *,
    bucket: str,
    gcs_prefix: str,
    project_name: str,
    experiment_id: str,
    snapshot_name: str,
    dry_run: bool = False,
    route_namespace: str = "",
) -> dict[str, str]:
    validate_eval_snapshot_name(snapshot_name)
    route_prefix = route_prefix_for(
        gcs_prefix=gcs_prefix,
        project_name=project_name,
        experiment_id=experiment_id,
        run_mode="dry_run" if dry_run else "full_run",
        route_namespace=route_namespace,
    )
    base = f"{route_prefix}/eval/snapshots/{snapshot_name}"
    return {
        "bucket": bucket,
        "base_path": base,
        "data_path": f"{base}/data.parquet",
        "manifest_path": f"{base}/manifest.json",
        "data_uri": f"gs://{bucket}/{base}/data.parquet",
        "manifest_uri": f"gs://{bucket}/{base}/manifest.json",
    }


def eval_augmentation_gcs_paths(
    *,
    bucket: str,
    gcs_prefix: str,
    project_name: str,
    experiment_id: str,
    snapshot_name: str,
    augmentation_dir: str,
    dry_run: bool = False,
    route_namespace: str = "",
) -> dict[str, str]:
    validate_eval_snapshot_name(snapshot_name)
    route_prefix = route_prefix_for(
        gcs_prefix=gcs_prefix,
        project_name=project_name,
        experiment_id=experiment_id,
        run_mode="dry_run" if dry_run else "full_run",
        route_namespace=route_namespace,
    )
    base = (
        f"{route_prefix}/eval/snapshots/{snapshot_name}"
        f"/augmentations/{augmentation_dir}"
    )
    return {
        "bucket": bucket,
        "base_path": base,
        "data_path": f"{base}/data.parquet",
        "manifest_path": f"{base}/manifest.json",
        "data_uri": f"gs://{bucket}/{base}/data.parquet",
        "manifest_uri": f"gs://{bucket}/{base}/manifest.json",
    }
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_eval_snapshot.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add automl/eval/snapshot.py tests/unit/test_eval_snapshot.py
git commit -m "Eval snapshot GCS path helpers"
```

---

### Task 7: Eval snapshot manifest builder + validator

**Files:**
- Modify: `automl/eval/snapshot.py`
- Test: `tests/unit/test_eval_snapshot.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_eval_snapshot.py`:

```python
def test_build_external_eval_manifest_round_trips():
    from automl.eval.snapshot import (
        build_eval_manifest, compute_eval_snapshot_identity,
        eval_snapshot_gcs_paths, validate_eval_manifest_v1,
    )

    df = _df_external()
    identity = compute_eval_snapshot_identity(
        kind="external", df=df, target_column="TARGET", hash_key="SK_ID_CURR",
    )
    paths = eval_snapshot_gcs_paths(
        bucket="b", gcs_prefix="automl",
        project_name="p", experiment_id="e",
        snapshot_name=identity.snapshot_name,
    )
    manifest = build_eval_manifest(
        identity=identity,
        project_name="p",
        experiment_id="e",
        paths=paths,
        df=df,
        provenance={"vintage": "2026Q2"},
    )

    assert manifest["schema_version"] == 1
    assert manifest["kind"] == "external"
    assert manifest["target_column"] == "TARGET"
    assert manifest["hash_key"] == ["SK_ID_CURR"]
    assert manifest["eval_snapshot_id"] == identity.snapshot_name
    assert manifest["hashes"]["eval_snapshot_hash"] == identity.eval_snapshot_hash
    assert manifest["gcs"]["data_uri"] == paths["data_uri"]
    assert manifest["provenance"] == {"vintage": "2026Q2"}
    assert "split_view" not in manifest

    validate_eval_manifest_v1(manifest, df=df)  # should not raise


def test_build_split_view_eval_manifest_round_trips():
    from automl.eval.snapshot import (
        build_eval_manifest, compute_eval_snapshot_identity,
        validate_eval_manifest_v1,
    )

    identity = compute_eval_snapshot_identity(
        kind="split_view",
        target_column="TARGET", hash_key="SK_ID_CURR",
        of_data_snapshot_id="sha256:abc", split_id_col="SPLITID",
        buckets=[(80, 100)],
    )
    manifest = build_eval_manifest(
        identity=identity,
        project_name="p",
        experiment_id="e",
        paths=None,
        df=None,
        provenance={},
    )

    assert manifest["kind"] == "split_view"
    assert manifest["split_view"]["of_data_snapshot_id"] == "sha256:abc"
    assert manifest["split_view"]["buckets"] == [[80, 100]]
    assert "gcs" not in manifest

    validate_eval_manifest_v1(manifest, df=None)  # should not raise


def test_validate_eval_manifest_v1_rejects_external_with_mismatched_hash():
    from automl.eval.snapshot import (
        build_eval_manifest, compute_eval_snapshot_identity,
        eval_snapshot_gcs_paths, validate_eval_manifest_v1,
    )

    df = _df_external()
    identity = compute_eval_snapshot_identity(
        kind="external", df=df, target_column="TARGET", hash_key="SK_ID_CURR",
    )
    paths = eval_snapshot_gcs_paths(
        bucket="b", gcs_prefix="automl",
        project_name="p", experiment_id="e",
        snapshot_name=identity.snapshot_name,
    )
    manifest = build_eval_manifest(
        identity=identity, project_name="p", experiment_id="e",
        paths=paths, df=df, provenance={},
    )

    mutated_df = df.copy()
    mutated_df.loc[0, "FEATURE_A"] = 999.0
    with pytest.raises(RuntimeError, match="content_hash"):
        validate_eval_manifest_v1(manifest, df=mutated_df)
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_eval_snapshot.py -v
```

Expected: 3 new FAIL — `build_eval_manifest`/`validate_eval_manifest_v1` don't exist.

- [ ] **Step 3: Implement the manifest builders**

Add to `automl/eval/snapshot.py`:

```python
from datetime import UTC, datetime


def build_eval_manifest(
    *,
    identity: EvalSnapshotIdentity,
    project_name: str,
    experiment_id: str,
    paths: dict[str, str] | None,
    df: pd.DataFrame | None,
    provenance: dict[str, Any],
    created_at: str | None = None,
) -> dict[str, Any]:
    if identity.kind == "external":
        if df is None or paths is None:
            raise ValueError("paths and df are required for kind='external'")
        if _frame_content_hash(df) != identity.content_hash:
            raise ValueError("df content_hash does not match identity")
    timestamp = created_at or datetime.now(UTC).isoformat()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "project_name": project_name,
        "experiment_id": experiment_id,
        "eval_snapshot_id": identity.snapshot_name,
        "kind": identity.kind,
        "target_column": identity.target_column,
        "hash_key": identity.hash_key,
        "created_at": timestamp,
        "hashes": {
            "eval_snapshot_hash": identity.eval_snapshot_hash,
            "schema_hash": identity.schema_hash,
            "content_hash": identity.content_hash,
        },
        "provenance": dict(provenance),
    }
    if identity.kind == "external":
        manifest["shape"] = {
            "n_rows": int(len(df)),
            "n_columns": int(len(df.columns)),
        }
        manifest["gcs"] = {
            "data_uri": paths["data_uri"],
            "manifest_uri": paths["manifest_uri"],
        }
    else:  # split_view
        manifest["split_view"] = {
            "of_data_snapshot_id": identity.of_data_snapshot_id,
            "split_id_col": identity.split_id_col,
            "buckets": [list(b) for b in (identity.buckets or ())],
        }
    return manifest


def validate_eval_manifest_v1(
    manifest: dict[str, Any],
    df: pd.DataFrame | None,
) -> None:
    if manifest.get("schema_version") != 1:
        raise RuntimeError("eval snapshot manifest schema_version must be 1")
    kind = manifest.get("kind")
    if kind not in ("external", "split_view"):
        raise RuntimeError(f"unknown eval snapshot kind: {kind!r}")
    hashes = manifest.get("hashes") or {}
    if kind == "external":
        if df is None:
            raise RuntimeError("df required to validate external eval snapshot manifest")
        if hashes.get("content_hash") != _frame_content_hash(df):
            raise RuntimeError("eval snapshot manifest content_hash does not match df")
        if hashes.get("schema_hash") != _schema_hash(df):
            raise RuntimeError("eval snapshot manifest schema_hash does not match df")
        shape = manifest.get("shape") or {}
        if shape.get("n_rows") != len(df):
            raise RuntimeError("eval snapshot manifest n_rows does not match df")
        hash_key = manifest["hash_key"]
        if hash_key not in df.columns:
            raise RuntimeError(f"hash_key {hash_key!r} missing from df")
    else:  # split_view
        sv = manifest.get("split_view") or {}
        for key in ("of_data_snapshot_id", "split_id_col", "buckets"):
            if not sv.get(key):
                raise RuntimeError(f"split_view manifest missing {key!r}")
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_eval_snapshot.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add automl/eval/snapshot.py tests/unit/test_eval_snapshot.py
git commit -m "Eval snapshot manifest builder and validator"
```

---

### Task 8: Augmentation identity + manifest

**Files:**
- Modify: `automl/eval/snapshot.py`
- Test: `tests/unit/test_eval_snapshot.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_eval_snapshot.py`:

```python
AUG_NAME_RE_CASES = [
    ("ltv", True),
    ("ltv_60", True),
    ("LTV", False),
    ("3things", False),
    ("ltv-60", False),
    ("", False),
]


@pytest.mark.parametrize("name,valid", AUG_NAME_RE_CASES)
def test_augmentation_name_validation(name, valid):
    from automl.eval.snapshot import validate_augmentation_name
    if valid:
        validate_augmentation_name(name)
    else:
        with pytest.raises(ValueError):
            validate_augmentation_name(name)


def test_compute_augmentation_identity_is_deterministic():
    from automl.eval.snapshot import compute_augmentation_identity

    df = pd.DataFrame({"SK_ID_CURR": [101, 102, 103], "LTV": [0.7, 0.8, 0.6]})
    a = compute_augmentation_identity(
        eval_snapshot_id="v1_e8a4c102",
        name="ltv",
        df=df, hash_key="SK_ID_CURR",
    )
    b = compute_augmentation_identity(
        eval_snapshot_id="v1_e8a4c102",
        name="ltv",
        df=df, hash_key="SK_ID_CURR",
    )
    assert a.content_hash == b.content_hash
    assert a.hash8 == b.hash8
    assert a.augmentation_dir == f"ltv__{a.hash8}"


def test_build_augmentation_manifest_round_trips():
    from automl.eval.snapshot import (
        build_augmentation_manifest, compute_augmentation_identity,
        validate_augmentation_manifest_v1,
    )

    df = pd.DataFrame({"SK_ID_CURR": [101, 102], "LTV": [0.7, 0.8]})
    aug = compute_augmentation_identity(
        eval_snapshot_id="v1_e8a4c102",
        name="ltv",
        df=df, hash_key="SK_ID_CURR",
    )
    manifest = build_augmentation_manifest(
        identity=aug, df=df,
        source={"sql_path": "eval_sql/ltv.sql"},
    )
    assert manifest["schema_version"] == 1
    assert manifest["eval_snapshot_id"] == "v1_e8a4c102"
    assert manifest["name"] == "ltv"
    assert manifest["hash_key"] == ["SK_ID_CURR"]
    assert manifest["content_hash"] == aug.content_hash
    assert manifest["columns"] == [{"name": "LTV", "dtype": "float64"}]
    validate_augmentation_manifest_v1(manifest, df=df)
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_eval_snapshot.py -v
```

Expected: new tests FAIL.

- [ ] **Step 3: Implement augmentation helpers**

Add to `automl/eval/snapshot.py`:

```python
AUGMENTATION_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def validate_augmentation_name(name: str) -> None:
    if not AUGMENTATION_NAME_RE.fullmatch(name):
        raise ValueError(
            f"augmentation name {name!r} must match {AUGMENTATION_NAME_RE.pattern}"
        )


@dataclass(frozen=True)
class AugmentationIdentity:
    eval_snapshot_id: str
    name: str
    content_hash: str
    hash8: str
    augmentation_dir: str
    hash_key: tuple[str, ...]


def compute_augmentation_identity(
    *,
    eval_snapshot_id: str,
    name: str,
    df: pd.DataFrame,
    hash_key,                       # str | list[str] | tuple[str, ...]
) -> AugmentationIdentity:
    validate_augmentation_name(name)
    validate_eval_snapshot_name(eval_snapshot_id)
    hash_key_tuple = _normalize_hash_key(hash_key)
    hash_key_cols = list(hash_key_tuple)
    missing = [c for c in hash_key_cols if c not in df.columns]
    if missing:
        raise ValueError(f"hash_key column(s) {missing} not in aug columns")
    if df[hash_key_cols].duplicated().any():
        raise ValueError(f"hash_key {hash_key_cols!r} must be unique per row in aug")
    aug_cols = [c for c in df.columns if c not in hash_key_cols]
    if not aug_cols:
        raise ValueError("augmentation must add at least one column besides hash_key")
    content_hash = _frame_content_hash(df)
    hash8 = content_hash.removeprefix("sha256:")[:8]
    return AugmentationIdentity(
        eval_snapshot_id=eval_snapshot_id,
        name=name,
        content_hash=content_hash,
        hash8=hash8,
        augmentation_dir=f"{name}__{hash8}",
        hash_key=hash_key_tuple,
    )


def build_augmentation_manifest(
    *,
    identity: AugmentationIdentity,
    df: pd.DataFrame,
    source: dict[str, Any],
    created_at: str | None = None,
) -> dict[str, Any]:
    timestamp = created_at or datetime.now(UTC).isoformat()
    hash_key_cols = list(identity.hash_key)
    aug_columns = [c for c in df.columns if c not in hash_key_cols]
    return {
        "schema_version": 1,
        "eval_snapshot_id": identity.eval_snapshot_id,
        "name": identity.name,
        "hash8": identity.hash8,
        "hash_key": hash_key_cols,
        "columns": [
            {"name": c, "dtype": str(df[c].dtype)} for c in aug_columns
        ],
        "shape": {"n_rows": int(len(df)), "n_columns": int(len(aug_columns))},
        "content_hash": identity.content_hash,
        "created_at": timestamp,
        "source": dict(source),
    }


def validate_augmentation_manifest_v1(
    manifest: dict[str, Any],
    df: pd.DataFrame,
) -> None:
    if manifest.get("schema_version") != 1:
        raise RuntimeError("augmentation manifest schema_version must be 1")
    if manifest.get("content_hash") != _frame_content_hash(df):
        raise RuntimeError("augmentation manifest content_hash does not match df")
    hash_key = manifest.get("hash_key") or []
    missing = [c for c in hash_key if c not in df.columns]
    if missing:
        raise RuntimeError(f"hash_key column(s) {missing} missing from aug df")
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_eval_snapshot.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add automl/eval/snapshot.py tests/unit/test_eval_snapshot.py
git commit -m "Augmentation identity and manifest"
```

---

---

## Phase 3 — Publish API (the governance gate)

Build `automl/eval/publish.py` exporting `prepare_eval_snapshot`,
`prepare_eval_split_view`, and `prepare_eval_augmentation`. Each is
idempotent on content hash (no-op if the same content was already
published). Uses `automl/io/gcs.py` for writes.

### Task 9: `prepare_eval_snapshot` for `kind=external`

**Files:**
- Create: `automl/eval/publish.py`
- Test: `tests/unit/test_eval_publish.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_eval_publish.py`:

```python
from __future__ import annotations
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def _df() -> pd.DataFrame:
    return pd.DataFrame({
        "SK_ID_CURR": [101, 102, 103],
        "F": [1.0, 2.0, 3.0],
        "TARGET": [0, 1, 0],
    })


def _ctx() -> MagicMock:
    ctx = MagicMock()
    ctx.gcs_bucket = "bucket"
    ctx.gcs_prefix = "automl"
    ctx.project_name = "p"
    ctx.experiment_id = "e"
    return ctx


def test_prepare_eval_snapshot_external_publishes_and_returns_pointer():
    from automl.eval import publish

    with patch.object(publish, "gcs_blob_exists", return_value=False) as exists, \
         patch.object(publish, "write_df_to_gcs_as_parquet") as write_parquet, \
         patch.object(publish, "write_json_to_gcs") as write_json:

        pointer = publish.prepare_eval_snapshot(
            ctx=_ctx(),
            frame=_df(),
            target_col="TARGET",
            hash_key="SK_ID_CURR",
            provenance={"vintage": "Q2"},
            dry_run=False,
        )

        assert pointer.kind == "external"
        assert pointer.eval_snapshot_id.startswith("v1_")
        write_parquet.assert_called_once()
        write_json.assert_called_once()


def test_prepare_eval_snapshot_external_is_idempotent_on_content_hash():
    from automl.eval import publish

    with patch.object(publish, "gcs_blob_exists", return_value=True), \
         patch.object(publish, "write_df_to_gcs_as_parquet") as write_parquet, \
         patch.object(publish, "write_json_to_gcs") as write_json:

        pointer = publish.prepare_eval_snapshot(
            ctx=_ctx(), frame=_df(),
            target_col="TARGET", hash_key="SK_ID_CURR",
            provenance={}, dry_run=False,
        )
        assert pointer.cached is True
        write_parquet.assert_not_called()
        write_json.assert_not_called()
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_eval_publish.py -v
```

Expected: FAIL — module/`prepare_eval_snapshot` don't exist.

- [ ] **Step 3: Implement `prepare_eval_snapshot` for external**

Create `automl/eval/publish.py`:

```python
"""Eval snapshot and augmentation publishing API."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from automl.core.project_context import ProjectContext
from automl.data.pipeline import (
    gcs_blob_exists,
    write_df_to_gcs_as_parquet,
    write_json_to_gcs,
)
from automl.eval.snapshot import (
    build_augmentation_manifest,
    build_eval_manifest,
    compute_augmentation_identity,
    compute_eval_snapshot_identity,
    eval_augmentation_gcs_paths,
    eval_snapshot_gcs_paths,
    validate_augmentation_name,
)


@dataclass(frozen=True)
class EvalSnapshotPointer:
    eval_snapshot_id: str
    kind: str
    bucket: str
    base_path: str
    manifest_uri: str
    data_uri: str | None
    hash_key: str
    target_column: str
    cached: bool


def prepare_eval_snapshot(
    *,
    ctx: ProjectContext,
    frame: pd.DataFrame,
    target_col: str,
    hash_key: str,
    provenance: dict[str, Any] | None = None,
    dry_run: bool = False,
    route_namespace: str = "",
) -> EvalSnapshotPointer:
    identity = compute_eval_snapshot_identity(
        kind="external",
        df=frame,
        target_column=target_col,
        hash_key=hash_key,
    )
    paths = eval_snapshot_gcs_paths(
        bucket=ctx.gcs_bucket,
        gcs_prefix=ctx.gcs_prefix,
        project_name=ctx.project_name,
        experiment_id=ctx.experiment_id,
        snapshot_name=identity.snapshot_name,
        dry_run=dry_run,
        route_namespace=route_namespace,
    )
    manifest_exists = gcs_blob_exists(paths["bucket"], paths["manifest_path"])
    data_exists = gcs_blob_exists(paths["bucket"], paths["data_path"])
    if manifest_exists and data_exists:
        return EvalSnapshotPointer(
            eval_snapshot_id=identity.snapshot_name,
            kind="external",
            bucket=paths["bucket"],
            base_path=paths["base_path"],
            manifest_uri=paths["manifest_uri"],
            data_uri=paths["data_uri"],
            hash_key=hash_key,
            target_column=target_col,
            cached=True,
        )
    manifest = build_eval_manifest(
        identity=identity,
        project_name=ctx.project_name,
        experiment_id=ctx.experiment_id,
        paths=paths,
        df=frame,
        provenance=provenance or {},
    )
    write_df_to_gcs_as_parquet(frame, paths["bucket"], paths["data_path"])
    write_json_to_gcs(manifest, paths["bucket"], paths["manifest_path"])
    return EvalSnapshotPointer(
        eval_snapshot_id=identity.snapshot_name,
        kind="external",
        bucket=paths["bucket"],
        base_path=paths["base_path"],
        manifest_uri=paths["manifest_uri"],
        data_uri=paths["data_uri"],
        hash_key=hash_key,
        target_column=target_col,
        cached=False,
    )
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_eval_publish.py -v
```

Expected: both pass.

- [ ] **Step 5: Commit**

```bash
git add automl/eval/publish.py tests/unit/test_eval_publish.py
git commit -m "prepare_eval_snapshot for kind=external with idempotent publish"
```

---

### Task 10: `prepare_eval_split_view` (pointer only — no GCS data write)

**Files:**
- Modify: `automl/eval/publish.py`
- Modify: `tests/unit/test_eval_publish.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_eval_publish.py`:

```python
def test_prepare_eval_split_view_writes_manifest_only_no_data():
    from automl.eval import publish

    with patch.object(publish, "gcs_blob_exists", return_value=False) as exists, \
         patch.object(publish, "write_df_to_gcs_as_parquet") as write_parquet, \
         patch.object(publish, "write_json_to_gcs") as write_json:

        pointer = publish.prepare_eval_split_view(
            ctx=_ctx(),
            data_snapshot_id="v3_a1b2c3d4",
            target_col="TARGET",
            hash_key="SK_ID_CURR",
            split_id_col="SPLITID",
            buckets=[(80, 100)],
            dry_run=False,
        )
        assert pointer.kind == "split_view"
        assert pointer.data_uri is None
        write_parquet.assert_not_called()
        write_json.assert_called_once()


def test_prepare_eval_split_view_is_idempotent_when_manifest_exists():
    from automl.eval import publish

    with patch.object(publish, "gcs_blob_exists", return_value=True), \
         patch.object(publish, "write_json_to_gcs") as write_json:
        pointer = publish.prepare_eval_split_view(
            ctx=_ctx(),
            data_snapshot_id="v3_a1b2c3d4",
            target_col="TARGET",
            hash_key="SK_ID_CURR",
            split_id_col="SPLITID",
            buckets=[(80, 100)],
            dry_run=False,
        )
        assert pointer.cached is True
        write_json.assert_not_called()
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_eval_publish.py -v
```

Expected: FAIL — `prepare_eval_split_view` not defined.

- [ ] **Step 3: Implement `prepare_eval_split_view`**

Add to `automl/eval/publish.py`:

```python
def _short_data_snapshot_id(data_snapshot_id: str) -> str:
    """Strip 'sha256:' prefix or 'v<n>_' prefix to keep manifest text compact."""
    if data_snapshot_id.startswith("sha256:"):
        return data_snapshot_id.removeprefix("sha256:")[:16]
    return data_snapshot_id


def prepare_eval_split_view(
    *,
    ctx: ProjectContext,
    data_snapshot_id: str,
    target_col: str,
    hash_key: str,
    split_id_col: str,
    buckets: list[tuple[int, int]],
    dry_run: bool = False,
    route_namespace: str = "",
) -> EvalSnapshotPointer:
    identity = compute_eval_snapshot_identity(
        kind="split_view",
        target_column=target_col,
        hash_key=hash_key,
        of_data_snapshot_id=data_snapshot_id,
        split_id_col=split_id_col,
        buckets=buckets,
    )
    paths = eval_snapshot_gcs_paths(
        bucket=ctx.gcs_bucket,
        gcs_prefix=ctx.gcs_prefix,
        project_name=ctx.project_name,
        experiment_id=ctx.experiment_id,
        snapshot_name=identity.snapshot_name,
        dry_run=dry_run,
        route_namespace=route_namespace,
    )
    if gcs_blob_exists(paths["bucket"], paths["manifest_path"]):
        return EvalSnapshotPointer(
            eval_snapshot_id=identity.snapshot_name,
            kind="split_view",
            bucket=paths["bucket"],
            base_path=paths["base_path"],
            manifest_uri=paths["manifest_uri"],
            data_uri=None,
            hash_key=hash_key,
            target_column=target_col,
            cached=True,
        )
    manifest = build_eval_manifest(
        identity=identity,
        project_name=ctx.project_name,
        experiment_id=ctx.experiment_id,
        paths=None,
        df=None,
        provenance={},
    )
    write_json_to_gcs(manifest, paths["bucket"], paths["manifest_path"])
    return EvalSnapshotPointer(
        eval_snapshot_id=identity.snapshot_name,
        kind="split_view",
        bucket=paths["bucket"],
        base_path=paths["base_path"],
        manifest_uri=paths["manifest_uri"],
        data_uri=None,
        hash_key=hash_key,
        target_column=target_col,
        cached=False,
    )
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_eval_publish.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add automl/eval/publish.py tests/unit/test_eval_publish.py
git commit -m "prepare_eval_split_view writes manifest pointer only"
```

---

### Task 11: `prepare_eval_augmentation` with the 8 publish rules

**Files:**
- Modify: `automl/eval/publish.py`
- Modify: `tests/unit/test_eval_publish.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_eval_publish.py`:

```python
def _aug_df_ok() -> pd.DataFrame:
    return pd.DataFrame({"SK_ID_CURR": [101, 102, 103], "LTV": [0.7, 0.8, 0.6]})


def _base_eval_manifest() -> dict:
    return {
        "schema_version": 1,
        "eval_snapshot_id": "v1_e8a4c102",
        "kind": "external",
        "target_column": "TARGET",
        "hash_key": ["SK_ID_CURR"],
        "shape": {"n_rows": 3, "n_columns": 3},
    }


def _base_eval_columns() -> list[str]:
    return ["SK_ID_CURR", "F", "TARGET"]


def _existing_aug_columns() -> dict[str, list[str]]:
    """Map of existing aug_dir → declared columns."""
    return {}


def test_prepare_eval_augmentation_publishes_when_rules_pass():
    from automl.eval import publish

    with patch.object(publish, "gcs_blob_exists", return_value=False), \
         patch.object(publish, "write_df_to_gcs_as_parquet") as wp, \
         patch.object(publish, "write_json_to_gcs") as wj, \
         patch.object(publish, "_load_eval_manifest_for_aug",
                      return_value=_base_eval_manifest()), \
         patch.object(publish, "_eval_base_row_ids",
                      return_value={101, 102, 103, 104}), \
         patch.object(publish, "_eval_base_columns",
                      return_value=_base_eval_columns()), \
         patch.object(publish, "_existing_augmentation_columns",
                      return_value=_existing_aug_columns()):

        pointer = publish.prepare_eval_augmentation(
            ctx=_ctx(),
            eval_snapshot_id="v1_e8a4c102",
            frame=_aug_df_ok(),
            name="ltv",
            source={"sql_path": "x.sql"},
        )
        assert pointer.name == "ltv"
        assert pointer.eval_snapshot_id == "v1_e8a4c102"
        wp.assert_called_once()
        wj.assert_called_once()


def test_prepare_eval_augmentation_refuses_override_of_base_column():
    from automl.eval import publish

    df = pd.DataFrame({"SK_ID_CURR": [101, 102], "F": [9.0, 9.0]})

    with patch.object(publish, "_load_eval_manifest_for_aug",
                      return_value=_base_eval_manifest()), \
         patch.object(publish, "_eval_base_row_ids",
                      return_value={101, 102}), \
         patch.object(publish, "_eval_base_columns",
                      return_value=_base_eval_columns()), \
         patch.object(publish, "_existing_augmentation_columns",
                      return_value=_existing_aug_columns()):

        with pytest.raises(ValueError, match="overlaps base column"):
            publish.prepare_eval_augmentation(
                ctx=_ctx(),
                eval_snapshot_id="v1_e8a4c102",
                frame=df, name="bad", source={},
            )


def test_prepare_eval_augmentation_refuses_overlap_with_other_aug():
    from automl.eval import publish

    df = pd.DataFrame({"SK_ID_CURR": [101, 102], "LTV": [0.7, 0.8]})
    existing = {"ltv_old__abc12345": ["LTV"]}

    with patch.object(publish, "_load_eval_manifest_for_aug",
                      return_value=_base_eval_manifest()), \
         patch.object(publish, "_eval_base_row_ids", return_value={101, 102}), \
         patch.object(publish, "_eval_base_columns",
                      return_value=_base_eval_columns()), \
         patch.object(publish, "_existing_augmentation_columns",
                      return_value=existing):
        with pytest.raises(ValueError, match="overlaps published augmentation"):
            publish.prepare_eval_augmentation(
                ctx=_ctx(), eval_snapshot_id="v1_e8a4c102",
                frame=df, name="ltv_new", source={},
            )


def test_prepare_eval_augmentation_refuses_orphan_rows():
    from automl.eval import publish

    df = pd.DataFrame({"SK_ID_CURR": [101, 999], "LTV": [0.7, 0.8]})
    with patch.object(publish, "_load_eval_manifest_for_aug",
                      return_value=_base_eval_manifest()), \
         patch.object(publish, "_eval_base_row_ids", return_value={101, 102}), \
         patch.object(publish, "_eval_base_columns",
                      return_value=_base_eval_columns()), \
         patch.object(publish, "_existing_augmentation_columns", return_value={}):
        with pytest.raises(ValueError, match="orphan"):
            publish.prepare_eval_augmentation(
                ctx=_ctx(), eval_snapshot_id="v1_e8a4c102",
                frame=df, name="ltv", source={},
            )


def test_prepare_eval_augmentation_is_idempotent_on_content_hash():
    from automl.eval import publish

    with patch.object(publish, "gcs_blob_exists", return_value=True), \
         patch.object(publish, "write_df_to_gcs_as_parquet") as wp, \
         patch.object(publish, "write_json_to_gcs") as wj, \
         patch.object(publish, "_load_eval_manifest_for_aug",
                      return_value=_base_eval_manifest()), \
         patch.object(publish, "_eval_base_row_ids",
                      return_value={101, 102, 103}), \
         patch.object(publish, "_eval_base_columns",
                      return_value=_base_eval_columns()), \
         patch.object(publish, "_existing_augmentation_columns", return_value={}):

        pointer = publish.prepare_eval_augmentation(
            ctx=_ctx(), eval_snapshot_id="v1_e8a4c102",
            frame=_aug_df_ok(), name="ltv", source={},
        )
        assert pointer.cached is True
        wp.assert_not_called()
        wj.assert_not_called()
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_eval_publish.py -v
```

Expected: 5 FAIL.

- [ ] **Step 3: Implement `prepare_eval_augmentation`**

Add to `automl/eval/publish.py`:

```python
@dataclass(frozen=True)
class AugmentationPointer:
    eval_snapshot_id: str
    name: str
    hash8: str
    augmentation_dir: str
    bucket: str
    base_path: str
    manifest_uri: str
    data_uri: str
    cached: bool


def _load_eval_manifest_for_aug(
    ctx: ProjectContext,
    eval_snapshot_id: str,
    dry_run: bool,
    route_namespace: str,
) -> dict[str, Any]:
    """Read the base eval snapshot manifest from GCS."""
    from automl.io.gcs import get_json_from_gcs
    paths = eval_snapshot_gcs_paths(
        bucket=ctx.gcs_bucket, gcs_prefix=ctx.gcs_prefix,
        project_name=ctx.project_name, experiment_id=ctx.experiment_id,
        snapshot_name=eval_snapshot_id, dry_run=dry_run,
        route_namespace=route_namespace,
    )
    return get_json_from_gcs(paths["bucket"], paths["manifest_path"])


def _eval_base_row_ids(
    ctx: ProjectContext,
    eval_manifest: dict[str, Any],
    dry_run: bool,
    route_namespace: str,
) -> set[tuple]:
    """Return the hash_key tuples of the base eval snapshot.

    Even for single-column hash_key the elements are 1-tuples — uniform
    set membership semantics across single/composite.
    """
    from automl.eval.loading import load_eval_snapshot
    snap = load_eval_snapshot(
        ctx=ctx, eval_snapshot_id=eval_manifest["eval_snapshot_id"],
        dry_run=dry_run, route_namespace=route_namespace,
    )
    rows = snap.df[list(snap.hash_key)].itertuples(index=False, name=None)
    return set(rows)


def _eval_base_columns(
    ctx: ProjectContext,
    eval_manifest: dict[str, Any],
    dry_run: bool,
    route_namespace: str,
) -> list[str]:
    from automl.eval.loading import load_eval_snapshot
    snap = load_eval_snapshot(
        ctx=ctx, eval_snapshot_id=eval_manifest["eval_snapshot_id"],
        dry_run=dry_run, route_namespace=route_namespace,
    )
    return list(snap.df.columns)


def _existing_augmentation_columns(
    ctx: ProjectContext,
    eval_snapshot_id: str,
    dry_run: bool,
    route_namespace: str,
) -> dict[str, list[str]]:
    """Map augmentation_dir → declared columns for all published augs on this eval."""
    from automl.io.gcs import gcs_list_prefixes, get_json_from_gcs
    paths = eval_snapshot_gcs_paths(
        bucket=ctx.gcs_bucket, gcs_prefix=ctx.gcs_prefix,
        project_name=ctx.project_name, experiment_id=ctx.experiment_id,
        snapshot_name=eval_snapshot_id, dry_run=dry_run,
        route_namespace=route_namespace,
    )
    aug_root = f"{paths['base_path']}/augmentations/"
    out: dict[str, list[str]] = {}
    for aug_dir in gcs_list_prefixes(paths["bucket"], aug_root):
        manifest_path = f"{aug_dir.rstrip('/')}/manifest.json"
        try:
            m = get_json_from_gcs(paths["bucket"], manifest_path)
        except Exception:
            continue
        dir_name = aug_dir.rstrip("/").split("/")[-1]
        out[dir_name] = [c["name"] for c in m.get("columns", [])]
    return out


def prepare_eval_augmentation(
    *,
    ctx: ProjectContext,
    eval_snapshot_id: str,
    frame: pd.DataFrame,
    name: str,
    source: dict[str, Any] | None = None,
    dry_run: bool = False,
    route_namespace: str = "",
) -> AugmentationPointer:
    validate_augmentation_name(name)
    eval_manifest = _load_eval_manifest_for_aug(
        ctx, eval_snapshot_id, dry_run, route_namespace
    )
    hash_key = tuple(eval_manifest["hash_key"])   # JSON list → tuple
    hash_key_cols = list(hash_key)

    # Rule 1: hash_key columns present and unique tuple in aug
    missing = [c for c in hash_key_cols if c not in frame.columns]
    if missing:
        raise ValueError(f"augmentation missing hash_key column(s): {missing}")
    if frame[hash_key_cols].duplicated().any():
        raise ValueError(f"hash_key {hash_key_cols!r} tuples must be unique in aug")

    aug_cols = [c for c in frame.columns if c not in hash_key_cols]
    if not aug_cols:
        raise ValueError("augmentation must add at least one column")

    # Rule 2: additive-only — no overlap with base columns
    base_cols = set(_eval_base_columns(ctx, eval_manifest, dry_run, route_namespace))
    overlap_base = set(aug_cols) & base_cols
    if overlap_base:
        raise ValueError(f"augmentation overlaps base column(s): {sorted(overlap_base)}")

    # Rule 3: no overlap with previously-published aug columns
    existing = _existing_augmentation_columns(ctx, eval_snapshot_id, dry_run, route_namespace)
    claimed: dict[str, str] = {}
    for aug_dir, cols in existing.items():
        for col in cols:
            claimed[col] = aug_dir
    overlap_other = set(aug_cols) & set(claimed)
    if overlap_other:
        first = sorted(overlap_other)[0]
        raise ValueError(
            f"augmentation column {first!r} overlaps published augmentation "
            f"{claimed[first]!r}"
        )

    # Rule 4: subset of base row id tuples
    base_row_ids = _eval_base_row_ids(ctx, eval_manifest, dry_run, route_namespace)
    aug_ids = set(frame[hash_key_cols].itertuples(index=False, name=None))
    orphans = aug_ids - base_row_ids
    if orphans:
        raise ValueError(
            f"augmentation has {len(orphans)} orphan row ids not present in base eval"
        )

    # Compute identity, check idempotency, write.
    identity = compute_augmentation_identity(
        eval_snapshot_id=eval_snapshot_id,
        name=name, df=frame, hash_key=hash_key,
    )
    paths = eval_augmentation_gcs_paths(
        bucket=ctx.gcs_bucket, gcs_prefix=ctx.gcs_prefix,
        project_name=ctx.project_name, experiment_id=ctx.experiment_id,
        snapshot_name=eval_snapshot_id,
        augmentation_dir=identity.augmentation_dir,
        dry_run=dry_run, route_namespace=route_namespace,
    )
    if gcs_blob_exists(paths["bucket"], paths["manifest_path"]) and \
       gcs_blob_exists(paths["bucket"], paths["data_path"]):
        return AugmentationPointer(
            eval_snapshot_id=eval_snapshot_id, name=name,
            hash8=identity.hash8, augmentation_dir=identity.augmentation_dir,
            bucket=paths["bucket"], base_path=paths["base_path"],
            manifest_uri=paths["manifest_uri"], data_uri=paths["data_uri"],
            cached=True,
        )
    manifest = build_augmentation_manifest(
        identity=identity, df=frame, source=source or {},
    )
    write_df_to_gcs_as_parquet(frame, paths["bucket"], paths["data_path"])
    write_json_to_gcs(manifest, paths["bucket"], paths["manifest_path"])
    return AugmentationPointer(
        eval_snapshot_id=eval_snapshot_id, name=name,
        hash8=identity.hash8, augmentation_dir=identity.augmentation_dir,
        bucket=paths["bucket"], base_path=paths["base_path"],
        manifest_uri=paths["manifest_uri"], data_uri=paths["data_uri"],
        cached=False,
    )
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_eval_publish.py -v
```

Expected: all pass. (If `gcs_list_prefixes` doesn't exist in `automl/io/gcs.py` yet, add a thin wrapper — see next task.)

- [ ] **Step 5: Commit**

```bash
git add automl/eval/publish.py tests/unit/test_eval_publish.py
git commit -m "prepare_eval_augmentation with publish-time validation rules"
```

---

### Task 12: GCS `list_prefixes` helper (if not already present)

**Files:**
- Modify: `automl/io/gcs.py`
- Test: `tests/unit/test_data_movers.py` (existing GCS helper tests live here)

- [ ] **Step 1: Check whether the helper exists**

```bash
grep -n "def gcs_list_prefixes\|def list_prefixes" automl/io/gcs.py
```

If a function returning a list of direct child "directories" (common prefixes) under a given GCS prefix already exists, skip this task. Otherwise:

- [ ] **Step 2: Write the failing test**

Append to `tests/unit/test_data_movers.py`:

```python
def test_gcs_list_prefixes_returns_direct_children(monkeypatch):
    from automl.io import gcs

    fake_iter = [
        type("P", (), {"prefixes": ["a/x/", "a/y/", "a/z/"]})(),
    ]
    fake_blobs = type("Blobs", (), {"pages": fake_iter})()
    fake_client = type("C", (), {"list_blobs": lambda self, b, prefix, delimiter: fake_blobs})()
    monkeypatch.setattr(gcs, "_get_client", lambda: fake_client())

    out = gcs.gcs_list_prefixes("bucket", "a/")
    assert sorted(out) == ["a/x/", "a/y/", "a/z/"]
```

- [ ] **Step 3: Implement**

Add to `automl/io/gcs.py`:

```python
def gcs_list_prefixes(bucket: str, prefix: str) -> list[str]:
    """List immediate sub-prefixes (folders) under a GCS prefix using delimiter='/'."""
    client = _get_client()
    blobs = client.list_blobs(bucket, prefix=prefix, delimiter="/")
    out: list[str] = []
    for page in blobs.pages:
        out.extend(getattr(page, "prefixes", []))
    return out
```

(Adjust to actual `_get_client` name in this file — check by reading the top of `automl/io/gcs.py`.)

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_data_movers.py -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add automl/io/gcs.py tests/unit/test_data_movers.py
git commit -m "gcs_list_prefixes helper for augmentation discovery"
```

---

## Phase 4 — Eval snapshot loading

`load_eval_snapshot` rehydrates either kind of eval snapshot: `external` reads its parquet; `split_view` rehydrates from the referenced data snapshot via the existing data-pipeline split helpers.

### Task 13: `load_eval_snapshot` for `external` kind

**Files:**
- Create: `automl/eval/loading.py`
- Test: `tests/unit/test_eval_loading.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_eval_loading.py`:

```python
from __future__ import annotations
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def _ctx() -> MagicMock:
    ctx = MagicMock()
    ctx.gcs_bucket = "bucket"
    ctx.gcs_prefix = "automl"
    ctx.project_name = "p"
    ctx.experiment_id = "e"
    return ctx


def _external_manifest_and_df():
    df = pd.DataFrame({
        "SK_ID_CURR": [101, 102, 103],
        "F": [1.0, 2.0, 3.0],
        "TARGET": [0, 1, 0],
    })
    from automl.eval.snapshot import (
        compute_eval_snapshot_identity, eval_snapshot_gcs_paths, build_eval_manifest,
    )
    identity = compute_eval_snapshot_identity(
        kind="external", df=df, target_column="TARGET", hash_key="SK_ID_CURR",
    )
    paths = eval_snapshot_gcs_paths(
        bucket="bucket", gcs_prefix="automl", project_name="p", experiment_id="e",
        snapshot_name=identity.snapshot_name,
    )
    manifest = build_eval_manifest(
        identity=identity, project_name="p", experiment_id="e",
        paths=paths, df=df, provenance={},
    )
    return manifest, df, identity.snapshot_name


def test_load_external_eval_snapshot_returns_df_and_metadata():
    from automl.eval.loading import load_eval_snapshot

    manifest, df, snapshot_id = _external_manifest_and_df()

    with patch("automl.eval.loading.get_json_from_gcs", return_value=manifest), \
         patch("automl.eval.loading.read_parquet_from_gcs", return_value=df):
        snap = load_eval_snapshot(
            ctx=_ctx(), eval_snapshot_id=snapshot_id, dry_run=False,
        )
    assert snap.kind == "external"
    assert snap.hash_key == "SK_ID_CURR"
    assert snap.target_column == "TARGET"
    assert list(snap.df.columns) == ["SK_ID_CURR", "F", "TARGET"]


def test_load_external_eval_snapshot_validates_content():
    from automl.eval.loading import load_eval_snapshot

    manifest, df, snapshot_id = _external_manifest_and_df()
    mutated = df.copy()
    mutated.loc[0, "F"] = 999.0

    with patch("automl.eval.loading.get_json_from_gcs", return_value=manifest), \
         patch("automl.eval.loading.read_parquet_from_gcs", return_value=mutated):
        with pytest.raises(RuntimeError, match="content_hash"):
            load_eval_snapshot(ctx=_ctx(), eval_snapshot_id=snapshot_id, dry_run=False)
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_eval_loading.py -v
```

Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement `load_eval_snapshot` for external**

Create `automl/eval/loading.py`:

```python
"""Load eval snapshots (external + split_view) from GCS."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from automl.core.project_context import ProjectContext
from automl.io.gcs import get_json_from_gcs, read_parquet_from_gcs
from automl.eval.snapshot import (
    eval_snapshot_gcs_paths,
    validate_eval_manifest_v1,
)


@dataclass(frozen=True)
class LoadedEvalSnapshot:
    df: pd.DataFrame
    kind: str
    eval_snapshot_id: str
    target_column: str
    hash_key: tuple[str, ...]                # always a tuple of column names
    manifest: dict[str, Any]
    row_ids: pd.DataFrame = field(default_factory=pd.DataFrame)
    # row_ids is the hash_key projection of df — one column per hash_key element.
    # For single-col hash_key, this is a 1-col DataFrame; for composite, multi-col.


def load_eval_snapshot(
    *,
    ctx: ProjectContext,
    eval_snapshot_id: str,
    dry_run: bool = False,
    route_namespace: str = "",
) -> LoadedEvalSnapshot:
    paths = eval_snapshot_gcs_paths(
        bucket=ctx.gcs_bucket, gcs_prefix=ctx.gcs_prefix,
        project_name=ctx.project_name, experiment_id=ctx.experiment_id,
        snapshot_name=eval_snapshot_id, dry_run=dry_run,
        route_namespace=route_namespace,
    )
    manifest = get_json_from_gcs(paths["bucket"], paths["manifest_path"])
    kind = manifest.get("kind")
    hash_key_tuple = tuple(manifest["hash_key"])    # JSON list → tuple
    if kind == "external":
        df = read_parquet_from_gcs(paths["bucket"], paths["data_path"])
        validate_eval_manifest_v1(manifest, df=df)
        return LoadedEvalSnapshot(
            df=df, kind="external",
            eval_snapshot_id=manifest["eval_snapshot_id"],
            target_column=manifest["target_column"],
            hash_key=hash_key_tuple,
            manifest=manifest,
            row_ids=df[list(hash_key_tuple)].reset_index(drop=True),
        )
    raise NotImplementedError(f"kind={kind!r} not yet supported")  # next task
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_eval_loading.py -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add automl/eval/loading.py tests/unit/test_eval_loading.py
git commit -m "load_eval_snapshot for kind=external"
```

---

### Task 14: `load_eval_snapshot` for `split_view` kind

**Files:**
- Modify: `automl/eval/loading.py`
- Modify: `tests/unit/test_eval_loading.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_eval_loading.py`:

```python
def test_load_split_view_eval_snapshot_rehydrates_from_data_snapshot():
    from automl.eval.loading import load_eval_snapshot

    full_df = pd.DataFrame({
        "SK_ID_CURR": [1, 2, 3, 4, 5],
        "F": [10, 20, 30, 40, 50],
        "TARGET": [0, 1, 0, 1, 0],
        "SPLITID": [10, 30, 75, 85, 95],
    })

    split_view_manifest = {
        "schema_version": 1,
        "eval_snapshot_id": "v1_abcd1234",
        "kind": "split_view",
        "target_column": "TARGET",
        "hash_key": ["SK_ID_CURR"],
        "split_view": {
            "of_data_snapshot_id": "v3_a1b2c3d4",
            "split_id_col": "SPLITID",
            "buckets": [[80, 100]],
        },
    }

    fake_data_snapshot = type("S", (), {
        "df_data": full_df,
        "manifest": {"target_column": "TARGET", "hash_key": ["SK_ID_CURR"]},
    })()

    with patch("automl.eval.loading.get_json_from_gcs", return_value=split_view_manifest), \
         patch("automl.eval.loading._load_data_snapshot_by_id",
               return_value=fake_data_snapshot):
        snap = load_eval_snapshot(
            ctx=_ctx(), eval_snapshot_id="v1_abcd1234", dry_run=False,
        )
    assert snap.kind == "split_view"
    assert list(snap.df["SK_ID_CURR"]) == [4, 5]  # SPLITID in [80, 100)
    assert snap.hash_key == "SK_ID_CURR"
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_eval_loading.py -v
```

Expected: FAIL — `kind=split_view` raises `NotImplementedError`.

- [ ] **Step 3: Implement split_view loading**

Replace the `raise NotImplementedError(...)` in `load_eval_snapshot` with:

```python
    if kind == "split_view":
        sv = manifest["split_view"]
        data_snap = _load_data_snapshot_by_id(
            ctx=ctx,
            data_snapshot_id=sv["of_data_snapshot_id"],
            dry_run=dry_run,
            route_namespace=route_namespace,
        )
        split_id_col = sv["split_id_col"]
        buckets = [tuple(b) for b in sv["buckets"]]
        mask = pd.Series(False, index=data_snap.df_data.index)
        for lo, hi in buckets:
            mask = mask | data_snap.df_data[split_id_col].between(lo, hi - 1)
        df = data_snap.df_data[mask].reset_index(drop=True)
        return LoadedEvalSnapshot(
            df=df, kind="split_view",
            eval_snapshot_id=manifest["eval_snapshot_id"],
            target_column=manifest["target_column"],
            hash_key=hash_key_tuple,
            manifest=manifest,
            row_ids=df[list(hash_key_tuple)].reset_index(drop=True),
        )
    raise RuntimeError(f"unknown eval snapshot kind: {kind!r}")
```

And add a private helper at the bottom of `automl/eval/loading.py`:

```python
def _load_data_snapshot_by_id(
    *,
    ctx: ProjectContext,
    data_snapshot_id: str,
    dry_run: bool,
    route_namespace: str,
):
    """Load the data snapshot frame referenced by `of_data_snapshot_id`.

    Goes through the data pipeline's existing snapshot loader so all
    integrity checks fire. We do NOT mirror the active-pointer logic
    because eval snapshots reference a specific historical snapshot id.
    """
    from automl.data.snapshot import snapshot_gcs_paths
    from automl.data.pipeline import (
        gcs_blob_exists, read_parquet_from_gcs as _read_parquet,
        get_csv_from_gcs, get_json_from_gcs as _read_json,
    )
    paths = snapshot_gcs_paths(
        bucket=ctx.gcs_bucket, gcs_prefix=ctx.gcs_prefix,
        project_name=ctx.project_name, experiment_id=ctx.experiment_id,
        snapshot_name=data_snapshot_id, dry_run=dry_run,
        route_namespace=route_namespace,
    )
    manifest = _read_json(paths["bucket"], paths["manifest_path"])
    df_data = _read_parquet(paths["bucket"], paths["data_path"])
    return type("DS", (), {"df_data": df_data, "manifest": manifest})()
```

Note: `_load_data_snapshot_by_id` returns a duck-typed object on purpose — we only need `.df_data` and `.manifest`. Don't import `LoadedDataSnapshot` here to avoid a circular dep.

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_eval_loading.py -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add automl/eval/loading.py tests/unit/test_eval_loading.py
git commit -m "load_eval_snapshot for kind=split_view rehydrates from data snapshot"
```

---

---

## Phase 5 — Predictions writer (GCS + manifest)

Replace today's MLflow + duplicate-GCS prediction writes with a single
GCS-keyed writer that puts `predictions/<eval_snapshot_id>/<trial_run_id>.parquet`
plus a JSON manifest at the same prefix.

### Task 15: `write_predictions_gcs` with manifest JSON

**Files:**
- Create: `automl/mlflow/artifacts/predictions.py`
- Test: `tests/unit/test_predictions_writer.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_predictions_writer.py`:

```python
from __future__ import annotations
import json
from unittest.mock import patch

import numpy as np
import pandas as pd


def test_write_predictions_gcs_writes_parquet_and_manifest():
    from automl.mlflow.artifacts.predictions import write_predictions_gcs

    captured: dict = {}

    def fake_write_parquet(df, bucket, object_name):
        captured.setdefault("parquets", []).append((bucket, object_name, df))

    def fake_write_json(payload, bucket, object_name):
        captured.setdefault("jsons", []).append((bucket, object_name, payload))

    id_frame = pd.DataFrame({"SK_ID_CURR": [101, 102, 103]})

    with patch("automl.mlflow.artifacts.predictions.write_df_to_gcs_as_parquet",
               side_effect=fake_write_parquet), \
         patch("automl.mlflow.artifacts.predictions.write_json_to_gcs",
               side_effect=fake_write_json):
        out = write_predictions_gcs(
            bucket="bucket",
            base_uri="gs://bucket/automl/p/e/predictions/v1_e8a4c102",
            trial_run_id="abc123",
            eval_snapshot_id="v1_e8a4c102",
            eval_snapshot_kind="split_view",
            label="test",
            hash_key=("SK_ID_CURR",),
            id_frame=id_frame,
            y_pred=np.array([0.1, 0.7, 0.4]),
            y_proba=None,
            augmentations_used=[],
        )

    assert out.predictions_uri.endswith("predictions/v1_e8a4c102/abc123.parquet")
    assert out.manifest_uri.endswith("predictions/v1_e8a4c102/abc123.json")

    parquet_bucket, parquet_path, parquet_df = captured["parquets"][0]
    assert list(parquet_df.columns) == ["SK_ID_CURR", "y_pred"]
    assert len(parquet_df) == 3

    json_bucket, json_path, manifest = captured["jsons"][0]
    assert manifest["schema_version"] == 1
    assert manifest["trial_run_id"] == "abc123"
    assert manifest["eval_snapshot_id"] == "v1_e8a4c102"
    assert manifest["label"] == "test"
    assert manifest["row_count"] == 3
    assert manifest["hash_key"] == ["SK_ID_CURR"]
    assert manifest["augmentations_used"] == []


def test_write_predictions_gcs_supports_composite_hash_key():
    from automl.mlflow.artifacts.predictions import write_predictions_gcs

    captured: dict = {}

    id_frame = pd.DataFrame({
        "CUST_ID": [1, 1, 2],
        "AS_OF":   ["Q1", "Q2", "Q1"],
    })

    with patch("automl.mlflow.artifacts.predictions.write_df_to_gcs_as_parquet",
               side_effect=lambda df, b, o: captured.setdefault("df", df)), \
         patch("automl.mlflow.artifacts.predictions.write_json_to_gcs",
               side_effect=lambda p, b, o: captured.setdefault("manifest", p)):
        write_predictions_gcs(
            bucket="b", base_uri="gs://b/x/predictions/v1_aa",
            trial_run_id="r", eval_snapshot_id="v1_aa",
            eval_snapshot_kind="external", label="test",
            hash_key=("AS_OF", "CUST_ID"),
            id_frame=id_frame,
            y_pred=np.array([0.1, 0.2, 0.3]),
            augmentations_used=[],
        )
    df = captured["df"]
    # hash_key columns come first, in canonical (sorted) order
    assert list(df.columns)[:2] == ["AS_OF", "CUST_ID"]
    assert "y_pred" in df.columns
    assert captured["manifest"]["hash_key"] == ["AS_OF", "CUST_ID"]


def test_write_predictions_gcs_with_y_proba_multiclass():
    from automl.mlflow.artifacts.predictions import write_predictions_gcs

    captured: dict = {}
    id_frame = pd.DataFrame({"ID": [1, 2]})

    with patch("automl.mlflow.artifacts.predictions.write_df_to_gcs_as_parquet",
               side_effect=lambda df, b, o: captured.setdefault("df", df)), \
         patch("automl.mlflow.artifacts.predictions.write_json_to_gcs"):
        write_predictions_gcs(
            bucket="b", base_uri="gs://b/x/predictions/v1_aa",
            trial_run_id="r", eval_snapshot_id="v1_aa",
            eval_snapshot_kind="external", label="test",
            hash_key=("ID",), id_frame=id_frame,
            y_pred=np.array([0, 1]),
            y_proba=np.array([[0.7, 0.3], [0.2, 0.8]]),
            class_labels=["class_a", "class_b"],
            augmentations_used=[],
        )
    df = captured["df"]
    assert "y_proba_class_a" in df.columns
    assert "y_proba_class_b" in df.columns
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_predictions_writer.py -v
```

Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement the writer**

Create `automl/mlflow/artifacts/predictions.py`:

```python
"""Prediction artifact writer — GCS only, keyed by eval_snapshot_id."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Sequence
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from automl.io.gcs import write_df_to_gcs_as_parquet, write_json_to_gcs


@dataclass(frozen=True)
class PredictionsArtifact:
    predictions_uri: str
    manifest_uri: str
    row_count: int


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "gs":
        raise ValueError(f"expected gs:// uri, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def write_predictions_gcs(
    *,
    bucket: str,
    base_uri: str,
    trial_run_id: str,
    eval_snapshot_id: str,
    eval_snapshot_kind: str,
    label: str,
    hash_key: tuple[str, ...],
    id_frame: pd.DataFrame,                  # rows × hash_key columns
    y_pred: Sequence[Any] | np.ndarray | pd.Series,
    y_proba: np.ndarray | None = None,
    class_labels: list[Any] | None = None,
    augmentations_used: list[dict[str, str]] | None = None,
) -> PredictionsArtifact:
    """Write predictions parquet + manifest JSON under predictions/<eval>/<run>.

    `id_frame` carries the hash_key columns for each row. For composite
    hash_key, this is a multi-column DataFrame; for single-column, it's a
    one-column DataFrame. Columns must match `hash_key` exactly (sorted).
    """
    parsed_bucket, base_path = _parse_gcs_uri(base_uri)
    if parsed_bucket != bucket:
        raise ValueError(
            f"bucket {bucket!r} does not match base_uri bucket {parsed_bucket!r}"
        )
    base_path = base_path.rstrip("/")
    parquet_path = f"{base_path}/{trial_run_id}.parquet"
    manifest_path = f"{base_path}/{trial_run_id}.json"

    hash_key_cols = list(hash_key)
    missing = [c for c in hash_key_cols if c not in id_frame.columns]
    if missing:
        raise ValueError(f"id_frame missing hash_key column(s): {missing}")

    y_pred_arr = np.asarray(list(y_pred))
    if len(id_frame) != len(y_pred_arr):
        raise ValueError("id_frame and y_pred must have the same length")

    df = id_frame[hash_key_cols].reset_index(drop=True).copy()
    df["y_pred"] = y_pred_arr
    if y_proba is not None:
        proba = np.asarray(y_proba)
        if proba.ndim == 1 or (proba.ndim == 2 and proba.shape[1] == 1):
            df["y_proba"] = proba.reshape(-1)
        elif proba.ndim == 2:
            labels = class_labels or list(range(proba.shape[1]))
            if len(labels) != proba.shape[1]:
                raise ValueError("class_labels length must match probability columns")
            for index, label_value in enumerate(labels):
                safe = str(label_value).strip().replace(" ", "_") or str(index)
                df[f"y_proba_{safe}"] = proba[:, index]
        else:
            raise ValueError(f"unsupported y_proba ndim: {proba.ndim}")

    write_df_to_gcs_as_parquet(df, bucket, parquet_path)

    manifest = {
        "schema_version": 1,
        "trial_run_id": trial_run_id,
        "eval_snapshot_id": eval_snapshot_id,
        "eval_snapshot_kind": eval_snapshot_kind,
        "label": label,
        "hash_key": hash_key_cols,
        "row_count": int(len(df)),
        "augmentations_used": list(augmentations_used or []),
        "written_at": datetime.now(UTC).isoformat(),
    }
    write_json_to_gcs(manifest, bucket, manifest_path)

    return PredictionsArtifact(
        predictions_uri=f"gs://{bucket}/{parquet_path}",
        manifest_uri=f"gs://{bucket}/{manifest_path}",
        row_count=int(len(df)),
    )
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_predictions_writer.py -v
```

Expected: both pass.

- [ ] **Step 5: Commit**

```bash
git add automl/mlflow/artifacts/predictions.py tests/unit/test_predictions_writer.py
git commit -m "write_predictions_gcs with manifest JSON"
```

---

## Phase 6 — `EvalSpec` redesign + augmentation joining

Flatten storage so `primary` is a string pointer in `report.json`; keep
the familiar constructor (`primary=Auc()`, `metrics=[others]`). Add
`Metric.required_augmentations` and the left-join behavior at compute
time.

### Task 16: `Metric.required_augmentations` field

**Files:**
- Modify: `automl/eval/base.py`
- Test: `tests/unit/test_evaluation_spec.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_evaluation_spec.py`:

```python
def test_metric_default_required_augmentations_is_empty():
    from automl.eval.metrics import Auc
    assert Auc().required_augmentations == ()


def test_metric_subclass_can_declare_required_augmentations():
    from automl.eval.base import Metric

    class LtvLift(Metric):
        required_augmentations = ("ltv",)
        required_columns = ("LTV",)
        def compute(self, df_test, y_pred, target_col):
            return float(df_test["LTV"].mean())

    assert LtvLift().required_augmentations == ("ltv",)
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_evaluation_spec.py -v
```

Expected: FAIL — attribute missing.

- [ ] **Step 3: Add the attribute**

In `automl/eval/base.py`, add to the `Metric` class:

```python
class Metric:
    name: str | None = None
    required_columns: tuple[str, ...] = ()
    required_augmentations: tuple[str, ...] = ()   # NEW
    _sign: int = 1
    _alias: str | None = None
    # ... rest unchanged
```

- [ ] **Step 4: Run the test**

```bash
uv run pytest tests/unit/test_evaluation_spec.py -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add automl/eval/base.py tests/unit/test_evaluation_spec.py
git commit -m "Metric.required_augmentations declarative field"
```

---

### Task 17: `EvalSpec` storage redesign (primary as pointer)

**Files:**
- Modify: `automl/eval/base.py`
- Test: `tests/unit/test_evaluation_spec.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_evaluation_spec.py`:

```python
def test_evalspec_evaluate_returns_flat_metrics_with_primary_pointer():
    from automl.eval import EvalSpec
    from automl.eval.metrics import Auc, LogLoss
    import pandas as pd

    spec = EvalSpec(primary=Auc(), metrics=[LogLoss()])
    df = pd.DataFrame({"TARGET": [0, 1, 1, 0]})
    result = spec.evaluate(df, [0.1, 0.9, 0.8, 0.2], "TARGET")

    assert result["primary"] == "auc"
    names = [m["name"] for m in result["metrics"]]
    assert names == ["auc", "log_loss"]
    auc_record = next(m for m in result["metrics"] if m["name"] == "auc")
    assert "value" in auc_record and isinstance(auc_record["value"], float)


def test_evalspec_with_only_primary_is_valid():
    from automl.eval import EvalSpec
    from automl.eval.metrics import Auc
    spec = EvalSpec(primary=Auc())
    assert spec.primary_name == "auc"
    assert len(spec.metrics) == 1   # primary is included internally


def test_evalspec_primary_must_not_collide_with_metrics():
    import pytest
    from automl.eval import EvalSpec
    from automl.eval.metrics import Auc
    with pytest.raises(ValueError, match="duplicate"):
        EvalSpec(primary=Auc(), metrics=[Auc()])
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_evaluation_spec.py -v
```

Expected: new tests FAIL — current `EvalSpec.evaluate` returns
`{"primary": {"name", "value"}, "metrics": [...]}` instead of the
flat-list + pointer shape.

- [ ] **Step 3: Update `EvalSpec.evaluate`**

In `automl/eval/base.py`, replace `EvalSpec.evaluate`:

```python
def evaluate(self, df_test: pd.DataFrame, y_pred: Any, target_col: str) -> dict[str, Any]:
    self.validate_columns(df_test, target_col)
    records: list[dict[str, Any]] = []
    for metric in (self._primary, *self._metrics):
        record = metric.evaluate(df_test, y_pred, target_col)
        record["augmentations"] = list(metric.required_augmentations)
        records.append(record)
    primary_value = records[0]["value"]
    if not is_scalar_value(primary_value):
        raise TypeError(
            f"primary metric '{records[0]['name']}' must return a scalar value"
        )
    records[0]["value"] = float(primary_value)
    return {
        "primary": records[0]["name"],   # POINTER now, not nested object
        "metrics": records,
    }
```

Also update `scalar_metric_records` in the same file — it currently expects
`primary` as a dict; change to handle the flat list:

```python
def scalar_metric_records(result: Mapping[str, Any]) -> dict[str, float]:
    scalars: dict[str, float] = {}
    primary_name = str(result.get("primary") or "")
    for record in result.get("metrics", []):
        if not isinstance(record, Mapping):
            continue
        name = str(record.get("name") or "")
        value = record.get("value")
        if name and is_scalar_value(value):
            scalars[name] = float(value)
            if name == primary_name:
                scalars["primary"] = float(value)
    return scalars
```

- [ ] **Step 4: Update existing callers in the same change**

Search for readers of the old shape and update them:

```bash
grep -rn '"primary"\]\.\["value"\]\|primary"\]\["name"\]\|result\["primary"\]\[' automl tests
```

Patch each call site to use `result["primary"]` (a string) plus the
`metrics` list. The main caller is `automl/runner/_execute.py` lines
~753–757 (`primary_record = evaluation.get("primary")`); update it to:

```python
primary_name = str(evaluation.get("primary") or "")
primary_record = next(
    (m for m in evaluation.get("metrics", []) if m.get("name") == primary_name),
    None,
)
if primary_record is None:
    raise RuntimeError(f"primary metric {primary_name!r} not in metrics list")
primary_value = float(primary_record["value"])
```

`tests/unit/test_eval_results_writer.py` will fail in the same way —
update its sample payloads to use the new flat shape.

- [ ] **Step 5: Run all eval tests**

```bash
uv run pytest tests/unit/test_evaluation_spec.py tests/unit/test_eval_results_writer.py tests/unit/test_evaluate_run.py -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add automl/eval/base.py automl/runner/_execute.py tests/unit/test_evaluation_spec.py tests/unit/test_eval_results_writer.py
git commit -m "EvalSpec.evaluate returns flat metrics with primary string pointer"
```

---

### Task 18: Augmentation left-join at compute time

**Files:**
- Modify: `automl/eval/base.py`
- Test: `tests/unit/test_evaluation_spec.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_evaluation_spec.py`:

```python
def test_evalspec_left_joins_augmentations_before_compute():
    from automl.eval import EvalSpec
    from automl.eval.base import Metric
    import pandas as pd

    class NeedsLTV(Metric):
        required_augmentations = ("ltv",)
        required_columns = ("LTV",)
        def compute(self, df_test, y_pred, target_col):
            return float(df_test["LTV"].sum())

    base_df = pd.DataFrame({
        "SK_ID_CURR": [1, 2, 3],
        "TARGET":    [0, 1, 0],
    })
    aug_frames = {
        "ltv": pd.DataFrame({"SK_ID_CURR": [1, 2, 3], "LTV": [10.0, 20.0, 30.0]}),
    }

    spec = EvalSpec(primary=NeedsLTV())
    result = spec.evaluate(
        df_test=base_df, y_pred=[0.1, 0.9, 0.5], target_col="TARGET",
        augmentation_frames=aug_frames, hash_key="SK_ID_CURR",
    )
    assert result["metrics"][0]["value"] == 60.0


def test_evalspec_errors_when_required_augmentation_is_missing():
    import pytest
    from automl.eval import EvalSpec
    from automl.eval.base import Metric
    import pandas as pd

    class NeedsLTV(Metric):
        required_augmentations = ("ltv",)
        def compute(self, df_test, y_pred, target_col):
            return 0.0

    df = pd.DataFrame({"SK_ID_CURR": [1], "TARGET": [0]})
    with pytest.raises(ValueError, match="augmentation"):
        EvalSpec(primary=NeedsLTV()).evaluate(
            df_test=df, y_pred=[0.5], target_col="TARGET",
            augmentation_frames={}, hash_key="SK_ID_CURR",
        )
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_evaluation_spec.py -v
```

Expected: FAIL — `EvalSpec.evaluate` doesn't accept `augmentation_frames`.

- [ ] **Step 3: Extend `EvalSpec.evaluate`**

Replace `evaluate` in `automl/eval/base.py`:

```python
def evaluate(
    self,
    df_test: pd.DataFrame,
    y_pred: Any,
    target_col: str,
    *,
    augmentation_frames: Mapping[str, pd.DataFrame] | None = None,
    hash_key: str | None = None,
) -> dict[str, Any]:
    self.validate_columns(df_test, target_col)
    augmentation_frames = augmentation_frames or {}

    # Determine which augmentation names are needed across all metrics.
    needed_augs: set[str] = set()
    for metric in (self._primary, *self._metrics):
        for aug_name in metric.required_augmentations:
            needed_augs.add(aug_name)
    missing = needed_augs - set(augmentation_frames)
    if missing:
        raise ValueError(
            f"required augmentations missing: {sorted(missing)}"
        )

    # Left-join all needed augmentations on hash_key before compute.
    df_joined = df_test
    if needed_augs:
        if not hash_key:
            raise ValueError("hash_key is required when metrics declare augmentations")
        for aug_name in sorted(needed_augs):
            aug = augmentation_frames[aug_name]
            if hash_key not in aug.columns:
                raise ValueError(
                    f"augmentation {aug_name!r} missing hash_key {hash_key!r}"
                )
            df_joined = df_joined.merge(aug, on=hash_key, how="left")

    records: list[dict[str, Any]] = []
    for metric in (self._primary, *self._metrics):
        record = metric.evaluate(df_joined, y_pred, target_col)
        record["augmentations"] = list(metric.required_augmentations)
        records.append(record)
    primary_value = records[0]["value"]
    if not is_scalar_value(primary_value):
        raise TypeError(
            f"primary metric '{records[0]['name']}' must return a scalar value"
        )
    records[0]["value"] = float(primary_value)
    return {"primary": records[0]["name"], "metrics": records}
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_evaluation_spec.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add automl/eval/base.py tests/unit/test_evaluation_spec.py
git commit -m "EvalSpec.evaluate left-joins augmentations on hash_key"
```

---

## Phase 7 — Model-eval compatibility predicate

Tiny module that checks `model.required_input_columns ⊆ eval_columns`
with dtype match before scoring.

### Task 19: `check_model_eval_compatibility`

**Files:**
- Create: `automl/eval/compatibility.py`
- Test: `tests/unit/test_eval_compatibility.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_eval_compatibility.py`:

```python
from __future__ import annotations
import pandas as pd
import pytest


def _model_with_required(cols_with_dtypes: dict[str, str]):
    class FakeModel:
        @property
        def required_input_columns(self):
            return list(cols_with_dtypes.keys())
        def required_input_dtypes(self):
            return dict(cols_with_dtypes)
    return FakeModel()


def test_compatibility_passes_when_columns_and_dtypes_match():
    from automl.eval.compatibility import check_model_eval_compatibility
    model = _model_with_required({"F1": "float64", "F2": "int64"})
    eval_df = pd.DataFrame({"F1": [1.0], "F2": [1], "EXTRA": ["x"]})
    check_model_eval_compatibility(model=model, eval_df=eval_df)


def test_compatibility_fails_on_missing_column():
    from automl.eval.compatibility import check_model_eval_compatibility, ColumnMissing
    model = _model_with_required({"F1": "float64", "F2": "int64"})
    eval_df = pd.DataFrame({"F1": [1.0]})
    with pytest.raises(ColumnMissing) as excinfo:
        check_model_eval_compatibility(model=model, eval_df=eval_df)
    assert "F2" in str(excinfo.value)


def test_compatibility_fails_on_dtype_mismatch():
    from automl.eval.compatibility import check_model_eval_compatibility, DtypeMismatch
    model = _model_with_required({"F1": "float64"})
    eval_df = pd.DataFrame({"F1": ["x"]})  # object, not float64
    with pytest.raises(DtypeMismatch) as excinfo:
        check_model_eval_compatibility(model=model, eval_df=eval_df)
    assert "F1" in str(excinfo.value)
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_eval_compatibility.py -v
```

Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement**

Create `automl/eval/compatibility.py`:

```python
"""Model-eval compatibility predicate."""
from __future__ import annotations

import pandas as pd


class ColumnMissing(ValueError):
    pass


class DtypeMismatch(ValueError):
    pass


def _canonical_dtype(dtype) -> str:
    """String-normalize dtype across pandas versions."""
    return str(dtype)


def check_model_eval_compatibility(
    *,
    model,
    eval_df: pd.DataFrame,
) -> None:
    required_cols = list(model.required_input_columns)
    missing = [c for c in required_cols if c not in eval_df.columns]
    if missing:
        raise ColumnMissing(
            f"eval frame missing model input columns: {missing}"
        )
    if hasattr(model, "required_input_dtypes"):
        expected = model.required_input_dtypes()
        mismatched: list[str] = []
        for col, exp in expected.items():
            actual = _canonical_dtype(eval_df[col].dtype)
            if _canonical_dtype(exp) != actual:
                mismatched.append(f"{col}: expected {exp}, got {actual}")
        if mismatched:
            raise DtypeMismatch(
                "eval frame dtype mismatch: " + "; ".join(mismatched)
            )
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_eval_compatibility.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add automl/eval/compatibility.py tests/unit/test_eval_compatibility.py
git commit -m "Model-eval compatibility predicate"
```

---

## Phase 8 — `evaluate()` verb

The single public entry point. Orchestrates: load eval snapshot, load
augmentations, check compatibility, score (or reuse cached predictions),
run `EvalSpec`, write predictions + manifest + `report.json` + update
`eval/manifest.json`, log MLflow metrics with label namespacing and the
unprefixed primary metric for the primary_label.

Building it in three tasks: base flow (no cache), idempotency rules,
primary_label management.

### Task 20: `evaluate()` — base flow, no caching

**Files:**
- Create: `automl/eval/evaluate.py`
- Test: `tests/integration/test_evaluate.py`

- [ ] **Step 1: Write a failing integration test**

Create `tests/integration/test_evaluate.py`:

```python
from __future__ import annotations
from pathlib import Path
from unittest.mock import MagicMock, patch

import mlflow
import numpy as np
import pandas as pd
import pytest


def _spin_up_local_mlflow(tmp_path: Path) -> str:
    tracking_uri = f"file:{tmp_path / 'mlruns'}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("test-eval-verb")
    return tracking_uri


def _ctx(tmp_path: Path, tracking_uri: str) -> MagicMock:
    ctx = MagicMock()
    ctx.gcs_bucket = "bucket"
    ctx.gcs_prefix = "automl"
    ctx.project_name = "p"
    ctx.experiment_id = "e"
    ctx.mlflow_tracking_uri = tracking_uri
    return ctx


def _fake_model():
    class M:
        @property
        def required_input_columns(self):
            return ["F1"]
        def required_input_dtypes(self):
            return {"F1": "float64"}
        def predict(self, df):
            return np.full(len(df), 0.5)
    return M()


def _external_eval_snapshot():
    df = pd.DataFrame({
        "SK_ID_CURR": [101, 102, 103],
        "F1": [1.0, 2.0, 3.0],
        "TARGET": [0, 1, 0],
    })
    from automl.eval.snapshot import compute_eval_snapshot_identity
    identity = compute_eval_snapshot_identity(
        kind="external", df=df, target_column="TARGET", hash_key="SK_ID_CURR",
    )
    manifest = {
        "schema_version": 1, "eval_snapshot_id": identity.snapshot_name,
        "kind": "external", "target_column": "TARGET",
        "hash_key": ["SK_ID_CURR"],
        "hashes": {
            "eval_snapshot_hash": identity.eval_snapshot_hash,
            "schema_hash": identity.schema_hash,
            "content_hash": identity.content_hash,
        },
        "shape": {"n_rows": 3, "n_columns": 3},
        "provenance": {},
    }
    return df, manifest, identity.snapshot_name


def test_evaluate_writes_predictions_report_and_metrics(tmp_path):
    tracking_uri = _spin_up_local_mlflow(tmp_path)
    ctx = _ctx(tmp_path, tracking_uri)

    df, manifest, eval_id = _external_eval_snapshot()

    with mlflow.start_run() as run:
        trial_run_id = run.info.run_id

    from automl.eval import EvalSpec
    from automl.eval.metrics import Auc, LogLoss

    captured: dict = {}

    def fake_load_model(run_id, project, project_root):
        return _fake_model()

    def fake_load_snapshot(*, ctx, eval_snapshot_id, dry_run, route_namespace=""):
        from automl.eval.loading import LoadedEvalSnapshot
        return LoadedEvalSnapshot(
            df=df, kind="external", eval_snapshot_id=eval_id,
            target_column="TARGET", hash_key="SK_ID_CURR",
            manifest=manifest, row_ids=df[["SK_ID_CURR"]].reset_index(drop=True),
        )

    def fake_write_predictions(**kwargs):
        captured.setdefault("predictions", []).append(kwargs)
        from automl.mlflow.artifacts.predictions import PredictionsArtifact
        return PredictionsArtifact(
            predictions_uri=f"gs://bucket/predictions/{eval_id}/{trial_run_id}.parquet",
            manifest_uri=f"gs://bucket/predictions/{eval_id}/{trial_run_id}.json",
            row_count=3,
        )

    with patch("automl.eval.evaluate.load_eval_snapshot", side_effect=fake_load_snapshot), \
         patch("automl.eval.evaluate.load_model", side_effect=fake_load_model), \
         patch("automl.eval.evaluate.write_predictions_gcs",
               side_effect=fake_write_predictions), \
         patch("automl.eval.evaluate._read_eval_report", return_value=None), \
         patch("automl.eval.evaluate._read_eval_manifest_toc", return_value=None), \
         patch("automl.eval.evaluate._write_eval_artifacts_to_mlflow"), \
         patch("automl.eval.evaluate._project_eval_spec",
               return_value=EvalSpec(primary=Auc(), metrics=[LogLoss()])):
        from automl.eval.evaluate import evaluate
        result = evaluate(
            ctx=ctx,
            model_run_id=trial_run_id,
            eval_snapshot_id=eval_id,
            label="test",
            dry_run=True,
        )

    assert result.cached is False
    assert result.label == "test"
    assert "auc" in result.metrics
    assert captured["predictions"][0]["label"] == "test"
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/integration/test_evaluate.py -v
```

Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement base `evaluate`**

Create `automl/eval/evaluate.py`:

```python
"""The single evaluate verb."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from automl.core.project_context import ProjectContext, resolve_project_context
from automl.eval.base import EvalSpec, scalar_metric_records
from automl.eval.compatibility import check_model_eval_compatibility
from automl.eval.loader import load_evaluation_spec
from automl.eval.loading import LoadedEvalSnapshot, load_eval_snapshot
from automl.inspect.views import load_model
from automl.mlflow.artifacts.predictions import write_predictions_gcs
from automl.mlflow.artifacts.gcs_paths import bucket_uri_for


@dataclass(frozen=True)
class EvaluateResult:
    trial_run_id: str
    eval_snapshot_id: str
    label: str
    predictions_uri: str
    metrics: dict[str, float]
    primary_metric: str
    is_primary_label: bool
    cached: bool
    mlflow_url: str = ""


def _project_eval_spec(ctx: ProjectContext) -> EvalSpec:
    return load_evaluation_spec(ctx)


def _resolve_label(label: str | None, eval_snapshot_id: str) -> str:
    if label:
        return label
    short = eval_snapshot_id.split("_")[-1][:8]
    return f"eval_{short}"


def _predictions_base_uri(
    ctx: ProjectContext, eval_snapshot_id: str, dry_run: bool
) -> str:
    """Return gs://bucket/<route>/predictions/<eval_id>"""
    from automl.mlflow.artifacts.gcs_paths import route_prefix_for
    prefix = route_prefix_for(
        gcs_prefix=ctx.gcs_prefix,
        project_name=ctx.project_name,
        experiment_id=ctx.experiment_id,
        run_mode="dry_run" if dry_run else "full_run",
    )
    return f"gs://{ctx.gcs_bucket}/{prefix}/predictions/{eval_snapshot_id}"


def _read_eval_report(client: MlflowClient, run_id: str, label: str) -> dict[str, Any] | None:
    """Return existing eval/<label>/report.json if it exists; else None."""
    import json
    try:
        local = client.download_artifacts(run_id, f"eval/{label}/report.json")
        return json.loads(Path(local).read_text())
    except Exception:
        return None


def _read_eval_manifest_toc(client: MlflowClient, run_id: str) -> dict[str, Any] | None:
    import json
    try:
        local = client.download_artifacts(run_id, "eval/manifest.json")
        return json.loads(Path(local).read_text())
    except Exception:
        return None


def _write_eval_artifacts_to_mlflow(
    client: MlflowClient,
    run_id: str,
    label: str,
    report: dict[str, Any],
    toc: dict[str, Any],
) -> None:
    import json, tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        report_dir = Path(tmp) / "eval" / label
        report_dir.mkdir(parents=True)
        (report_dir / "report.json").write_text(json.dumps(report, indent=2))
        client.log_artifacts(run_id, str(Path(tmp) / "eval" / label),
                             artifact_path=f"eval/{label}")
        toc_dir = Path(tmp) / "eval_toc"
        toc_dir.mkdir(parents=True)
        (toc_dir / "manifest.json").write_text(json.dumps(toc, indent=2))
        client.log_artifact(run_id, str(toc_dir / "manifest.json"),
                            artifact_path="eval")


def evaluate(
    *,
    ctx: ProjectContext | None = None,
    model_run_id: str,
    eval_snapshot_id: str,
    eval_spec: EvalSpec | None = None,
    label: str | None = None,
    overwrite: bool = False,
    set_as_primary_label: bool = False,
    project: str | None = None,
    project_root: Path | None = None,
    dry_run: bool = False,
    route_namespace: str = "",
) -> EvaluateResult:
    if ctx is None:
        ctx = resolve_project_context(project_root, project, start=Path.cwd())
    label_resolved = _resolve_label(label, eval_snapshot_id)
    spec = eval_spec or _project_eval_spec(ctx)

    tracking_uri = ctx.mlflow_tracking_uri
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri or None)

    eval_snap = load_eval_snapshot(
        ctx=ctx, eval_snapshot_id=eval_snapshot_id,
        dry_run=dry_run, route_namespace=route_namespace,
    )

    model = load_model(model_run_id, project=ctx.project_name, project_root=ctx.repo_root)
    feature_only = eval_snap.df.drop(columns=[eval_snap.target_column])
    check_model_eval_compatibility(model=model, eval_df=feature_only)

    y_pred = model.predict(feature_only)

    # Score against EvalSpec; no augmentations in base flow yet.
    eval_result = spec.evaluate(
        df_test=eval_snap.df, y_pred=y_pred, target_col=eval_snap.target_column,
    )

    # Write predictions to GCS.
    pred_base = _predictions_base_uri(ctx, eval_snapshot_id, dry_run)
    pred_artifact = write_predictions_gcs(
        bucket=ctx.gcs_bucket,
        base_uri=pred_base,
        trial_run_id=model_run_id,
        eval_snapshot_id=eval_snapshot_id,
        eval_snapshot_kind=eval_snap.kind,
        label=label_resolved,
        hash_key=eval_snap.hash_key,
        id_frame=eval_snap.df[list(eval_snap.hash_key)],
        y_pred=y_pred,
        augmentations_used=[],
    )

    # Build per-eval report.
    report = {
        "schema_version": 2,
        "label": label_resolved,
        "eval_snapshot_id": eval_snapshot_id,
        "eval_snapshot_kind": eval_snap.kind,
        "predictions_uri": pred_artifact.predictions_uri,
        "augmentations_used": [],
        "primary": eval_result["primary"],
        "metrics": eval_result["metrics"],
        "computed_at": datetime.now(UTC).isoformat(),
    }

    # Update toc.
    toc = _read_eval_manifest_toc(client, model_run_id) or {
        "schema_version": 1, "primary_label": None, "evaluations": [],
    }
    toc_entries = [e for e in toc["evaluations"] if e["label"] != label_resolved]
    toc_entries.append({
        "label": label_resolved,
        "eval_snapshot_id": eval_snapshot_id,
        "kind": eval_snap.kind,
        "primary_metric": eval_result["primary"],
        "computed_at": report["computed_at"],
    })
    toc["evaluations"] = sorted(toc_entries, key=lambda e: e["label"])
    if set_as_primary_label:
        toc["primary_label"] = label_resolved

    _write_eval_artifacts_to_mlflow(client, model_run_id, label_resolved, report, toc)

    # Log MLflow scalar metrics namespaced by label.
    scalars = scalar_metric_records(eval_result)
    scalars.pop("primary", None)
    for name, value in scalars.items():
        client.log_metric(model_run_id, f"{label_resolved}.{name}", float(value))
    # Primary label: log unprefixed primary metric.
    if toc["primary_label"] == label_resolved:
        primary_value = next(
            m["value"] for m in eval_result["metrics"] if m["name"] == eval_result["primary"]
        )
        client.log_metric(model_run_id, eval_result["primary"], float(primary_value))
        client.set_tag(model_run_id, "eval.primary_metric", eval_result["primary"])

    return EvaluateResult(
        trial_run_id=model_run_id,
        eval_snapshot_id=eval_snapshot_id,
        label=label_resolved,
        predictions_uri=pred_artifact.predictions_uri,
        metrics={m["name"]: float(m["value"]) for m in eval_result["metrics"]},
        primary_metric=eval_result["primary"],
        is_primary_label=(toc["primary_label"] == label_resolved),
        cached=False,
        mlflow_url="",
    )
```

Also export from `automl/eval/__init__.py`:

```python
from automl.eval.evaluate import evaluate, EvaluateResult  # noqa: F401
```

- [ ] **Step 4: Run the test**

```bash
uv run pytest tests/integration/test_evaluate.py -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add automl/eval/evaluate.py automl/eval/__init__.py tests/integration/test_evaluate.py
git commit -m "evaluate() base flow: score, write predictions, update report+toc"
```

---

### Task 21: `evaluate()` — idempotency / upsert / overwrite rules

**Files:**
- Modify: `automl/eval/evaluate.py`
- Modify: `tests/integration/test_evaluate.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/integration/test_evaluate.py`:

```python
def test_evaluate_returns_cached_result_when_report_already_exists(tmp_path):
    # Setup: existing report.json with auc=0.7 already logged for this (run, eval, label).
    tracking_uri = _spin_up_local_mlflow(tmp_path)
    ctx = _ctx(tmp_path, tracking_uri)
    df, manifest, eval_id = _external_eval_snapshot()

    with mlflow.start_run() as run:
        trial_run_id = run.info.run_id

    existing_report = {
        "schema_version": 2, "label": "test",
        "eval_snapshot_id": eval_id, "eval_snapshot_kind": "external",
        "predictions_uri": f"gs://bucket/predictions/{eval_id}/{trial_run_id}.parquet",
        "augmentations_used": [],
        "primary": "auc",
        "metrics": [
            {"name": "auc",     "value": 0.7, "augmentations": []},
            {"name": "log_loss","value": 0.4, "augmentations": []},
        ],
        "computed_at": "2026-05-18T00:00:00Z",
    }

    from automl.eval import EvalSpec
    from automl.eval.metrics import Auc, LogLoss
    spec = EvalSpec(primary=Auc(), metrics=[LogLoss()])

    with patch("automl.eval.evaluate.load_eval_snapshot"), \
         patch("automl.eval.evaluate.load_model"), \
         patch("automl.eval.evaluate.write_predictions_gcs") as wp, \
         patch("automl.eval.evaluate._read_eval_report", return_value=existing_report), \
         patch("automl.eval.evaluate._project_eval_spec", return_value=spec):
        from automl.eval.evaluate import evaluate
        result = evaluate(
            ctx=ctx, model_run_id=trial_run_id, eval_snapshot_id=eval_id,
            label="test", dry_run=True,
        )
    assert result.cached is True
    assert result.metrics["auc"] == 0.7
    wp.assert_not_called()


def test_evaluate_appends_new_metric_without_recomputing_existing(tmp_path):
    tracking_uri = _spin_up_local_mlflow(tmp_path)
    ctx = _ctx(tmp_path, tracking_uri)
    df, manifest, eval_id = _external_eval_snapshot()

    with mlflow.start_run() as run:
        trial_run_id = run.info.run_id

    existing_report = {
        "schema_version": 2, "label": "test",
        "eval_snapshot_id": eval_id, "eval_snapshot_kind": "external",
        "predictions_uri": f"gs://bucket/predictions/{eval_id}/{trial_run_id}.parquet",
        "augmentations_used": [],
        "primary": "auc",
        "metrics": [{"name": "auc", "value": 0.7, "augmentations": []}],
        "computed_at": "2026-05-18T00:00:00Z",
    }

    from automl.eval import EvalSpec
    from automl.eval.metrics import Auc, LogLoss
    new_spec = EvalSpec(primary=Auc(), metrics=[LogLoss()])

    with patch("automl.eval.evaluate.load_eval_snapshot",
               return_value=type("S", (), {
                   "df": df, "kind": "external", "hash_key": ("SK_ID_CURR",),
                   "target_column": "TARGET", "manifest": manifest,
                   "eval_snapshot_id": eval_id,
                   "row_ids": df[["SK_ID_CURR"]].reset_index(drop=True),
               })()), \
         patch("automl.eval.evaluate.load_model", return_value=_fake_model()), \
         patch("automl.eval.evaluate.write_predictions_gcs") as wp, \
         patch("automl.eval.evaluate._read_eval_report", return_value=existing_report), \
         patch("automl.eval.evaluate._read_eval_manifest_toc", return_value=None), \
         patch("automl.eval.evaluate._write_eval_artifacts_to_mlflow"), \
         patch("automl.eval.evaluate._project_eval_spec", return_value=new_spec):
        from automl.eval.evaluate import evaluate
        result = evaluate(
            ctx=ctx, model_run_id=trial_run_id, eval_snapshot_id=eval_id,
            label="test", dry_run=True,
        )
    assert result.cached is False
    assert result.metrics["auc"] == 0.7         # reused
    assert "log_loss" in result.metrics          # appended
    wp.assert_not_called()                       # predictions cached
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/integration/test_evaluate.py -v
```

Expected: 2 FAIL — base flow always re-scores.

- [ ] **Step 3: Add upsert logic**

Replace the body of `evaluate(...)` in `automl/eval/evaluate.py` after the
`eval_snap = load_eval_snapshot(...)` line and before predictions write
with:

```python
    existing = _read_eval_report(client, model_run_id, label_resolved)

    # Hard error: label-collision with different eval_snapshot_id unless overwrite.
    if existing and existing.get("eval_snapshot_id") != eval_snapshot_id and not overwrite:
        raise ValueError(
            f"label {label_resolved!r} already maps to "
            f"{existing['eval_snapshot_id']!r}; pass a different label or overwrite=True"
        )

    # Determine which metric names already have cached values.
    cached_by_name: dict[str, dict[str, Any]] = {}
    if existing:
        for record in existing.get("metrics", []):
            cached_by_name[record["name"]] = record

    requested_names = {m.resolved_name() for m in (spec._primary, *spec._metrics)}
    needs_new = requested_names - set(cached_by_name) if not overwrite else set()
    must_recompute = bool(overwrite) or (not existing) or bool(needs_new)

    if not must_recompute:
        # All metrics are cached. Return cached.
        metrics_dict = {n: float(r["value"]) for n, r in cached_by_name.items()}
        return EvaluateResult(
            trial_run_id=model_run_id,
            eval_snapshot_id=eval_snapshot_id,
            label=label_resolved,
            predictions_uri=str(existing.get("predictions_uri") or ""),
            metrics=metrics_dict,
            primary_metric=str(existing.get("primary") or ""),
            is_primary_label=False,
            cached=True,
        )

    # Reuse predictions if not overwriting AND we have an existing report.
    reuse_predictions = (existing is not None) and (not overwrite)

    model = load_model(model_run_id, project=ctx.project_name, project_root=ctx.repo_root)
    feature_only = eval_snap.df.drop(columns=[eval_snap.target_column])
    check_model_eval_compatibility(model=model, eval_df=feature_only)

    if reuse_predictions:
        # Skip prediction compute; we have the report's uri.
        y_pred = model.predict(feature_only)  # cheap path: trust cache via report
        pred_uri = str(existing.get("predictions_uri") or "")
    else:
        y_pred = model.predict(feature_only)
        pred_base = _predictions_base_uri(ctx, eval_snapshot_id, dry_run)
        pred_artifact = write_predictions_gcs(
            bucket=ctx.gcs_bucket, base_uri=pred_base,
            trial_run_id=model_run_id, eval_snapshot_id=eval_snapshot_id,
            eval_snapshot_kind=eval_snap.kind, label=label_resolved,
            hash_key=eval_snap.hash_key,
            id_frame=eval_snap.df[list(eval_snap.hash_key)],
            y_pred=y_pred, augmentations_used=[],
        )
        pred_uri = pred_artifact.predictions_uri

    if reuse_predictions:
        # Build a partial EvalSpec that only computes the missing names.
        from automl.eval.base import EvalSpec as _ES
        missing_metrics = [m for m in (spec._primary, *spec._metrics)
                           if m.resolved_name() in needs_new]
        if missing_metrics:
            partial_spec = _ES(primary=missing_metrics[0], metrics=missing_metrics[1:])
            partial_result = partial_spec.evaluate(
                df_test=eval_snap.df, y_pred=y_pred,
                target_col=eval_snap.target_column,
            )
            for record in partial_result["metrics"]:
                cached_by_name[record["name"]] = record
        # Primary points to what the new EvalSpec declares.
        primary_name = spec._primary.resolved_name()
    else:
        eval_result = spec.evaluate(
            df_test=eval_snap.df, y_pred=y_pred, target_col=eval_snap.target_column,
        )
        cached_by_name = {r["name"]: r for r in eval_result["metrics"]}
        primary_name = eval_result["primary"]

    final_metrics = list(cached_by_name.values())
    final_metrics.sort(key=lambda r: 0 if r["name"] == primary_name else 1)

    report = {
        "schema_version": 2,
        "label": label_resolved,
        "eval_snapshot_id": eval_snapshot_id,
        "eval_snapshot_kind": eval_snap.kind,
        "predictions_uri": pred_uri,
        "augmentations_used": [],
        "primary": primary_name,
        "metrics": final_metrics,
        "computed_at": datetime.now(UTC).isoformat(),
    }
    # (toc update + mlflow logs proceed as before)
```

The toc/mlflow-log block from Task 20 follows unchanged.

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/integration/test_evaluate.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add automl/eval/evaluate.py tests/integration/test_evaluate.py
git commit -m "evaluate() upsert: cache hits, metric append, overwrite rules"
```

---

### Task 22: `evaluate()` — augmentation resolution at score time

**Files:**
- Modify: `automl/eval/evaluate.py`
- Modify: `tests/integration/test_evaluate.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/integration/test_evaluate.py`:

```python
def test_evaluate_loads_and_joins_required_augmentations(tmp_path):
    tracking_uri = _spin_up_local_mlflow(tmp_path)
    ctx = _ctx(tmp_path, tracking_uri)
    df, manifest, eval_id = _external_eval_snapshot()

    with mlflow.start_run() as run:
        trial_run_id = run.info.run_id

    ltv_df = pd.DataFrame({"SK_ID_CURR": [101, 102, 103],
                           "LTV": [0.7, 0.8, 0.6]})

    from automl.eval.base import Metric
    from automl.eval import EvalSpec

    class LtvMean(Metric):
        required_augmentations = ("ltv",)
        required_columns = ("LTV",)
        def compute(self, df_test, y_pred, target_col):
            return float(df_test["LTV"].mean())

    spec = EvalSpec(primary=LtvMean())

    with patch("automl.eval.evaluate.load_eval_snapshot",
               return_value=type("S", (), {
                   "df": df, "kind": "external", "hash_key": ("SK_ID_CURR",),
                   "target_column": "TARGET", "manifest": manifest,
                   "eval_snapshot_id": eval_id,
                   "row_ids": df[["SK_ID_CURR"]].reset_index(drop=True),
               })()), \
         patch("automl.eval.evaluate.load_model", return_value=_fake_model()), \
         patch("automl.eval.evaluate.write_predictions_gcs"), \
         patch("automl.eval.evaluate._read_eval_report", return_value=None), \
         patch("automl.eval.evaluate._read_eval_manifest_toc", return_value=None), \
         patch("automl.eval.evaluate._write_eval_artifacts_to_mlflow"), \
         patch("automl.eval.evaluate._project_eval_spec", return_value=spec), \
         patch("automl.eval.evaluate._load_required_augmentations",
               return_value={"ltv": (ltv_df, "a3f1c204")}):
        from automl.eval.evaluate import evaluate
        result = evaluate(
            ctx=ctx, model_run_id=trial_run_id, eval_snapshot_id=eval_id,
            label="test", dry_run=True,
        )
    assert pytest.approx(result.metrics["ltv_mean"], 1e-9) == 0.7
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/integration/test_evaluate.py -v
```

Expected: FAIL — `_load_required_augmentations` doesn't exist and
`evaluate` doesn't pass augmentation frames to the spec.

- [ ] **Step 3: Add augmentation loader + plumb to spec**

Add to `automl/eval/evaluate.py`:

```python
def _load_required_augmentations(
    *,
    ctx: ProjectContext,
    eval_snapshot_id: str,
    augmentation_names: list[str],
    dry_run: bool,
    route_namespace: str,
) -> dict[str, tuple[pd.DataFrame, str]]:
    """Resolve aug names → (frame, hash8) for the latest published aug per name."""
    if not augmentation_names:
        return {}
    from automl.eval.snapshot import eval_snapshot_gcs_paths
    from automl.io.gcs import (
        gcs_list_prefixes, get_json_from_gcs, read_parquet_from_gcs,
    )
    paths = eval_snapshot_gcs_paths(
        bucket=ctx.gcs_bucket, gcs_prefix=ctx.gcs_prefix,
        project_name=ctx.project_name, experiment_id=ctx.experiment_id,
        snapshot_name=eval_snapshot_id, dry_run=dry_run,
        route_namespace=route_namespace,
    )
    aug_root = f"{paths['base_path']}/augmentations/"
    by_name: dict[str, tuple[pd.DataFrame, str, str]] = {}
    for aug_prefix in gcs_list_prefixes(paths["bucket"], aug_root):
        dir_name = aug_prefix.rstrip("/").split("/")[-1]
        name, _, hash8 = dir_name.partition("__")
        if name not in augmentation_names:
            continue
        manifest = get_json_from_gcs(paths["bucket"], f"{aug_prefix.rstrip('/')}/manifest.json")
        created = manifest.get("created_at", "")
        prior = by_name.get(name)
        if prior is None or created > prior[2]:
            df = read_parquet_from_gcs(paths["bucket"], f"{aug_prefix.rstrip('/')}/data.parquet")
            by_name[name] = (df, hash8, created)
    missing = sorted(set(augmentation_names) - set(by_name))
    if missing:
        raise ValueError(f"augmentations not published on eval snapshot: {missing}")
    return {name: (frame, hash8) for name, (frame, hash8, _) in by_name.items()}
```

In the score path of `evaluate(...)`, before calling `spec.evaluate(...)`,
add:

```python
needed_augs = sorted({
    aug for m in (spec._primary, *spec._metrics) for aug in m.required_augmentations
})
loaded_augs = _load_required_augmentations(
    ctx=ctx, eval_snapshot_id=eval_snapshot_id,
    augmentation_names=needed_augs,
    dry_run=dry_run, route_namespace=route_namespace,
)
augmentation_frames = {name: frame for name, (frame, _) in loaded_augs.items()}
augmentations_used = [{"name": n, "hash8": h} for n, (_, h) in loaded_augs.items()]
```

Then change `spec.evaluate(...)` invocations to pass:

```python
spec.evaluate(
    df_test=eval_snap.df, y_pred=y_pred,
    target_col=eval_snap.target_column,
    augmentation_frames=augmentation_frames,
    hash_key=eval_snap.hash_key,
)
```

And update the `report` dict's `augmentations_used` and the
`write_predictions_gcs(...)` `augmentations_used=` to use the new list.

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/integration/test_evaluate.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add automl/eval/evaluate.py tests/integration/test_evaluate.py
git commit -m "evaluate(): resolve and left-join required augmentations"
```

---

## Phase 9 — Trial-time integration

Replace the bespoke eval/prediction code in `automl/runner/_execute.py`
(~lines 740–830) with two `evaluate(...)` calls.

### Task 23: Rewrite `_execute.py` eval block to use `evaluate(...)` twice

**Files:**
- Modify: `automl/runner/_execute.py`
- Test: `tests/integration/test_runner_integration.py`

- [ ] **Step 1: Write a failing integration test**

Add to `tests/integration/test_runner_integration.py` (or its existing
equivalent):

```python
def test_trial_writes_test_and_train_evals_with_split_view_pointers(tmp_path, monkeypatch):
    # This test runs a small trial end-to-end against file-backed MLflow,
    # mocking GCS. Assert that:
    #   - eval/manifest.json has rows for label='train' and label='test'
    #   - eval/test/report.json exists with label='test', primary='auc'
    #   - eval/train/report.json exists with label='train'
    #   - MLflow scalar metrics 'test.auc' and 'train.auc' exist
    #   - MLflow scalar metric 'auc' (unprefixed primary) exists
    #   - No 'eval/predictions.parquet' is logged as an MLflow artifact
    ...
```

Flesh out the body using the existing pattern in `tests/integration/test_runner_integration.py`. Mock `prepare_eval_split_view` and `write_predictions_gcs` so no real GCS calls happen; let MLflow be file-backed.

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/integration/test_runner_integration.py::test_trial_writes_test_and_train_evals_with_split_view_pointers -v
```

Expected: FAIL.

- [ ] **Step 3: Rewrite the eval block in `_execute.py`**

In `automl/runner/_execute.py`, locate the block that writes predictions
and the eval report (today around lines 740–830). Replace with:

```python
# 4. Run eval via the unified evaluate verb (test + train as split_view pointers).
current_phase = "evaluation"
phase_started = time.monotonic()
from automl.eval import evaluate
from automl.eval.publish import prepare_eval_split_view

# Resolve a single load of the current data snapshot id from the loaded snapshot.
data_snapshot_id = snapshot.snapshot_name   # the human-readable v<n>_<hash8> form
hash_key = tuple(snapshot.manifest["hash_key"])
test_buckets = list(ctx.run_config.split.test_buckets())
train_buckets = list(ctx.run_config.split.train_buckets())

test_eval = prepare_eval_split_view(
    ctx=ctx, data_snapshot_id=data_snapshot_id,
    target_col=target_col, hash_key=hash_key,
    split_id_col=snapshot.manifest["split_id_col"],
    buckets=test_buckets, dry_run=dry_run,
)
train_eval = prepare_eval_split_view(
    ctx=ctx, data_snapshot_id=data_snapshot_id,
    target_col=target_col, hash_key=hash_key,
    split_id_col=snapshot.manifest["split_id_col"],
    buckets=train_buckets, dry_run=dry_run,
)

test_result = evaluate(
    ctx=ctx, model_run_id=_mlflow_run.info.run_id,
    eval_snapshot_id=test_eval.eval_snapshot_id,
    label="test", set_as_primary_label=True, dry_run=dry_run,
)
result["metrics"] = test_result.metrics
result["primary_metric_name"] = test_result.primary_metric
result["primary_metric_value"] = test_result.metrics[test_result.primary_metric]

try:
    evaluate(
        ctx=ctx, model_run_id=_mlflow_run.info.run_id,
        eval_snapshot_id=train_eval.eval_snapshot_id,
        label="train", dry_run=dry_run,
    )
except Exception as exc:
    log.warning("train-eval failed (best-effort): %s", exc)

timing.add("evaluation", time.monotonic() - phase_started)
```

Delete the old `write_predictions(...)` / `write_evaluation_results(...)`
block entirely. (Search for those names in `_execute.py` and confirm nothing
else remains.)

Note: the data snapshot manifest already carries `snapshot_identity_hash`
and `snapshot_name`; resolve `data_snapshot_id` to whichever the eval
snapshot identity expects (snapshot_name is the human-readable
`v<n>_<hash8>` form; use that).

- [ ] **Step 4: Run integration tests**

```bash
uv run pytest tests/integration/test_runner_integration.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add automl/runner/_execute.py tests/integration/test_runner_integration.py
git commit -m "Trial-time eval uses evaluate() for test and train split views"
```

---

## Phase 10 — Cleanups, contracts, CLI, notebooks

### Task 24: Delete `automl/reevaluation.py` and update imports

**Files:**
- Delete: `automl/reevaluation.py`
- Modify: `automl/__init__.py`
- Modify: any caller still importing `automl.reevaluation`

- [ ] **Step 1: Find all callers**

```bash
grep -rn "from automl.reevaluation\|automl\.reevaluate\|automl import.*reevaluate" automl tests projects
```

- [ ] **Step 2: Remove or rewrite each caller**

For each caller, replace `automl.reevaluate(run_id, df_eval=…, label=…)`
with the new flow:

```python
from automl.eval import evaluate
from automl.eval.publish import prepare_eval_snapshot

snap = prepare_eval_snapshot(
    ctx=ctx, frame=df_eval, target_col=target_col,
    hash_key=hash_key, provenance={...},
)
result = evaluate(
    ctx=ctx, model_run_id=run_id,
    eval_snapshot_id=snap.eval_snapshot_id,
    label="oot_q2_2026",
)
```

- [ ] **Step 3: Delete the file**

```bash
git rm automl/reevaluation.py
```

Also remove the `reevaluate` re-export from `automl/__init__.py` if present.

- [ ] **Step 4: Run the test suite**

```bash
uv run pytest tests/ -v -x
```

Expected: pass (all callers migrated).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Remove automl.reevaluation; replaced by evaluate verb"
```

---

### Task 25: Delete `write_predictions` and update `eval/report.json` writer

**Files:**
- Modify: `automl/mlflow/artifacts/eval.py`
- Modify: `automl/mlflow/artifacts/__init__.py`
- Modify: `automl/mlflow/__init__.py`
- Test: `tests/unit/test_eval_results_writer.py`

- [ ] **Step 1: Update the unit test**

In `tests/unit/test_eval_results_writer.py`, replace the test that asserts
`eval/report.json` lives at the top level with one that asserts the new
per-label path is written by `write_evaluation_results`:

```python
def test_writes_per_label_report(tmp_path: Path) -> None:
    from automl.mlflow.artifacts.eval import write_evaluation_results
    payload = {
        "schema_version": 2,
        "label": "test",
        "eval_snapshot_id": "v1_aaaa",
        "eval_snapshot_kind": "split_view",
        "predictions_uri": "gs://b/predictions/v1_aaaa/r.parquet",
        "augmentations_used": [],
        "primary": "auc",
        "metrics": [{"name": "auc", "value": 0.78, "augmentations": []}],
        "computed_at": "2026-05-18T00:00:00Z",
    }
    out = write_evaluation_results(tmp_path, payload)
    assert out == tmp_path / "eval" / "test" / "report.json"
```

Remove the test that asserts `write_predictions` behavior — that function
is deleted.

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_eval_results_writer.py -v
```

Expected: FAIL.

- [ ] **Step 3: Update `write_evaluation_results` and delete `write_predictions`**

Replace `automl/mlflow/artifacts/eval.py` with:

```python
"""Per-label evaluation report writer."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_SCHEMA_VERSION = 2


def write_evaluation_results(trial_dir: Path, payload: dict[str, Any]) -> Path:
    """Write per-eval `eval/<label>/report.json`."""
    label = payload.get("label")
    if not isinstance(label, str) or not label:
        raise ValueError("evaluation payload must contain 'label'")
    if "primary" not in payload:
        raise ValueError("evaluation payload must contain 'primary'")
    out_dir = trial_dir / "eval" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "report.json"
    document = {**payload, "schema_version": _SCHEMA_VERSION}
    out_path.write_text(json.dumps(document, indent=2, default=float))
    return out_path
```

Update `automl/mlflow/artifacts/__init__.py`:

```python
# Drop:
#   from automl.mlflow.artifacts.eval import write_evaluation_results, write_predictions
# Add:
from automl.mlflow.artifacts.eval import write_evaluation_results
from automl.mlflow.artifacts.predictions import write_predictions_gcs

__all__ = [
    ...,
    "write_evaluation_results",
    "write_predictions_gcs",
    # remove "write_predictions"
]
```

Same for `automl/mlflow/__init__.py` (the top-level re-export).

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_eval_results_writer.py tests/ -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add automl/mlflow/artifacts/eval.py automl/mlflow/artifacts/__init__.py automl/mlflow/__init__.py tests/unit/test_eval_results_writer.py
git commit -m "Replace write_predictions with GCS writer; per-label report path"
```

---

### Task 26: Update `inspect/views.py` for the new artifact tree

**Files:**
- Modify: `automl/inspect/views.py`
- Test: `tests/unit/test_inspect.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_inspect.py`:

```python
def test_leaderboard_reads_unprefixed_primary_metric():
    # The leaderboard sorts by the unprefixed scalar metric (the primary).
    # With the new design, that metric is logged for the primary_label.
    ...
    # Use existing test fixtures / mock client patterns from this file.


def test_show_trial_reads_eval_manifest_toc():
    # show_trial returns one entry per label found in eval/manifest.json.
    ...
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_inspect.py -v
```

Expected: FAIL.

- [ ] **Step 3: Update `inspect/views.py`**

Locate where show_trial reads `eval/report.json` (single file) and change
to:

1. Read `eval/manifest.json` (TOC).
2. For each `evaluations[i].label`, read `eval/<label>/report.json`.
3. Return a list of `EvalView(label=…, eval_snapshot_id=…, primary=…, metrics={…})`.

Leaderboard already reads the unprefixed primary scalar metric — no change
needed if the metric is logged as `<primary_name>` for the primary_label.

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_inspect.py -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add automl/inspect/views.py tests/unit/test_inspect.py
git commit -m "Inspect views read eval/<label>/report.json via manifest TOC"
```

---

### Task 27: Update `automl/loop_context/` proposer/coder render

**Files:**
- Modify: `automl/loop_context/` (the specific file that reads eval artifacts; identify via `grep -rn "eval/report" automl/loop_context`)
- Test: `tests/unit/test_context_*.py` (existing tests in the same path)

- [ ] **Step 1: Locate context render of eval data**

```bash
grep -rn "eval/report\|primary_metric\|primary metric" automl/loop_context tests/unit/test_context_*
```

- [ ] **Step 2: Write the failing test**

Append to whichever `tests/unit/test_context_*.py` covers proposer/coder
packet rendering:

```python
def test_proposer_packet_uses_current_primary_label_from_manifest():
    # When eval/manifest.json declares primary_label="test", proposer packet
    # surfaces test.auc (or its unprefixed equivalent) as the headline metric.
    ...
```

- [ ] **Step 3: Update context render**

Change the code to read `eval/manifest.json` for `primary_label` and
`primary_metric`, then read `eval/<primary_label>/report.json` for the
headline metric. The agent never reads "what was once primary" — only the
current pointer.

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_context_*.py -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add automl/loop_context/ tests/unit/test_context_*.py
git commit -m "Context render reads current primary_label from eval/manifest.json"
```

---

### Task 28: Add CLI verb `automl eval`

**Files:**
- Create: `automl/cli/eval.py`
- Modify: `automl/cli/__init__.py`
- Test: `tests/unit/test_cli_dispatch.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_cli_dispatch.py`:

```python
def test_cli_eval_verb_dispatches_to_evaluate(monkeypatch):
    from automl.cli import dispatch
    called: dict = {}
    def fake_evaluate(**kwargs):
        called.update(kwargs)
        class R:
            metrics = {"auc": 0.7}
            primary_metric = "auc"
            mlflow_url = ""
            predictions_uri = ""
            label = kwargs["label"]
            eval_snapshot_id = kwargs["eval_snapshot_id"]
            cached = False
        return R()
    monkeypatch.setattr("automl.eval.evaluate", fake_evaluate)
    exit_code = dispatch([
        "eval",
        "--model-run-id", "abc123",
        "--eval-snapshot-id", "v1_e8a4c102",
        "--label", "test",
    ])
    assert exit_code == 0
    assert called["model_run_id"] == "abc123"
    assert called["eval_snapshot_id"] == "v1_e8a4c102"
    assert called["label"] == "test"
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_cli_dispatch.py -v
```

Expected: FAIL — no `eval` verb registered.

- [ ] **Step 3: Implement the verb**

Create `automl/cli/eval.py`:

```python
"""`automl eval` — evaluate a model against a published eval snapshot."""
from __future__ import annotations

import argparse

from automl.cli import register


@register("eval", help="Evaluate a model run against a published eval snapshot")
def eval_main(parser: argparse.ArgumentParser, argv: list[str]) -> int:
    parser.add_argument("--model-run-id", required=True)
    parser.add_argument("--eval-snapshot-id", required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--set-as-primary-label", action="store_true")
    parser.add_argument("--project", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    from automl.eval import evaluate
    result = evaluate(
        model_run_id=args.model_run_id,
        eval_snapshot_id=args.eval_snapshot_id,
        label=args.label,
        overwrite=args.overwrite,
        set_as_primary_label=args.set_as_primary_label,
        project=args.project,
        dry_run=args.dry_run,
    )
    print(f"label={result.label}  eval={result.eval_snapshot_id}")
    print(f"primary={result.primary_metric}  metrics={result.metrics}")
    print(f"predictions={result.predictions_uri}")
    return 0
```

In `automl/cli/__init__.py`, add `"automl.cli.eval"` to `_SUBCOMMAND_MODULES`.

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_cli_dispatch.py -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add automl/cli/eval.py automl/cli/__init__.py tests/unit/test_cli_dispatch.py
git commit -m "automl eval CLI verb"
```

---

### Task 29: Update notebook `6_reevaluate_existing_model.ipynb`

**Files:**
- Modify: `projects/example_homecredit/notebooks/6_reevaluate_existing_model.ipynb`

- [ ] **Step 1: Update notebook cells**

Open `6_reevaluate_existing_model.ipynb` and replace its content with
cells equivalent to:

```python
# Cell 1 (markdown)
# # Re-evaluate Existing Model
#
# Score a logged model against a labeled eval snapshot (split view or external)
# and log the result on the trial run (no child runs).

# Cell 2
from pathlib import Path
import automl
from automl import inspect
from automl.data import build_pipeline
from automl.eval import evaluate
from automl.eval.publish import prepare_eval_split_view

DRY_RUN = True
ctx = automl.context()

leaderboard = inspect.leaderboard(dry_run=DRY_RUN, training_origin="all")
deployed = leaderboard[0]

# Cell 3
pipeline = build_pipeline(ctx, dry_run=DRY_RUN)
pipeline.prepare_data()
snapshot = pipeline.load_data_snapshot()

# Cell 4: rescore the deployed model against the test split of the LATEST data snapshot
test_eval = prepare_eval_split_view(
    ctx=ctx,
    data_snapshot_id=snapshot.snapshot_name,
    target_col=snapshot.manifest["target_column"],
    hash_key=snapshot.manifest["hash_key"],
    split_id_col=snapshot.manifest["split_id_col"],
    buckets=ctx.run_config.split.test_buckets(),
    dry_run=DRY_RUN,
)
result = evaluate(
    ctx=ctx,
    model_run_id=deployed.run_id,
    eval_snapshot_id=test_eval.eval_snapshot_id,
    label="test_latest_data",
    dry_run=DRY_RUN,
)
result.metrics, result.mlflow_url

# Cell 5
inspect.show_trial(run_id=deployed.run_id)["evaluations"]
```

- [ ] **Step 2: Verify executability (smoke)**

```bash
uv run jupyter nbconvert --to script projects/example_homecredit/notebooks/6_reevaluate_existing_model.ipynb --stdout | head
```

Expected: clean Python output.

- [ ] **Step 3: Commit**

```bash
git add projects/example_homecredit/notebooks/6_reevaluate_existing_model.ipynb
git commit -m "Notebook 6: rewrite to use evaluate + prepare_eval_split_view"
```

---

### Task 30: Contract tests for new shape

**Files:**
- Modify: `tests/contracts/test_package_organization.py`
- Create: `tests/contracts/test_eval_snapshot_layout.py`

- [ ] **Step 1: Write the new contract tests**

Create `tests/contracts/test_eval_snapshot_layout.py`:

```python
"""Ratchet tests pinning the new eval snapshot / predictions layout."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_eval_snapshot_module_exists():
    assert (ROOT / "automl" / "eval" / "snapshot.py").exists()


def test_eval_publish_module_exists():
    assert (ROOT / "automl" / "eval" / "publish.py").exists()


def test_evaluate_module_exists():
    assert (ROOT / "automl" / "eval" / "evaluate.py").exists()


def test_predictions_writer_module_exists():
    assert (ROOT / "automl" / "mlflow" / "artifacts" / "predictions.py").exists()


def test_reevaluation_module_is_removed():
    assert not (ROOT / "automl" / "reevaluation.py").exists()


def test_cli_eval_verb_registered():
    text = (ROOT / "automl" / "cli" / "__init__.py").read_text()
    assert '"automl.cli.eval"' in text


def test_evaluate_is_exported_from_eval_package():
    text = (ROOT / "automl" / "eval" / "__init__.py").read_text()
    assert "evaluate" in text
```

- [ ] **Step 2: Update `test_package_organization.py`** to remove any pin that
  references the removed paths (`automl/reevaluation.py`,
  `eval/predictions.parquet` MLflow artifact path, etc.).

- [ ] **Step 3: Run contracts**

```bash
uv run pytest tests/contracts/ -v
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add tests/contracts/
git commit -m "Contract tests pin new eval snapshot / predictions layout"
```

---

### Task 31: Regenerate regression goldens

**Files:**
- Modify: `tests/regression/*.json` (the golden manifests)

- [ ] **Step 1: Inspect existing goldens**

```bash
ls tests/regression
```

For each `.json` golden that captures trial run artifact structure, update
the expected file tree:

- Remove `eval/report.json` and `eval/predictions.parquet`.
- Add `eval/manifest.json`, `eval/test/report.json`, `eval/train/report.json`.

- [ ] **Step 2: Regenerate via the regression's normal update procedure**

If the regression tests use a `UPDATE_GOLDENS=1` env var or a similar
mechanism (check the test file), invoke it:

```bash
UPDATE_GOLDENS=1 uv run pytest tests/regression -v
```

Otherwise update the JSON files by hand to match the new artifact tree.

- [ ] **Step 3: Run regression**

```bash
uv run pytest tests/regression -v
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add tests/regression/
git commit -m "Regression: update goldens for new eval artifact tree"
```

---

### Task 32: End-to-end smoke test

**Files:**
- Modify or create: `tests/integration/test_eval_end_to_end.py`

- [ ] **Step 1: Write a comprehensive smoke test**

Create `tests/integration/test_eval_end_to_end.py`:

```python
"""End-to-end smoke: publish data snapshot → publish split-view eval →
score a trivial model → write predictions to GCS (mock) → run evaluate
twice (cached the second time) → publish an aug → re-eval with the aug
→ verify report contains the new metric."""
from __future__ import annotations
# (Adapt the existing tests/integration/test_runner_integration.py
#  fixtures for tmp MLflow + mocked GCS.)
```

Walk through the spec's "What this gives the user" section and assert
each one explicitly in this test. Use mocks for GCS (`patch` everything
under `automl.io.gcs`) so the test is hermetic.

- [ ] **Step 2: Run the test**

```bash
uv run pytest tests/integration/test_eval_end_to_end.py -v
```

Expected: pass.

- [ ] **Step 3: Run the full suite to catch any regression**

```bash
uv run pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_eval_end_to_end.py
git commit -m "End-to-end smoke test for eval snapshot + evaluate verb"
```

---

## Self-review

- **Spec coverage:** Every spec section maps to at least one task.
  - "Eval snapshot identity" → Tasks 4, 5
  - "hash_key invariant + uniqueness validation" → Tasks 1, 2 (Task 3 retired)
  - "GCS layout" → Tasks 6, 9–11, 15
  - "Augmentations" → Tasks 8, 11, 18, 22
  - "Public API surface" → Tasks 9–14, 20, 28
  - "Idempotency semantics" → Task 21
  - "EvalSpec redesign" → Tasks 16–18
  - "Model-eval compatibility predicate" → Task 19
  - "Trial-time integration" → Task 23
  - "Trial run artifact tree" → Tasks 20, 25, 26
  - "Code surface" deletes → Tasks 24, 25
  - "CLI verb" → Task 28
  - "Notebook updates" → Task 29
  - "Tests" → Tasks 30, 31, 32

- **Placeholder scan:** Tasks 26, 27, 31, 32 leave some test bodies as
  "fill in based on existing fixtures" (`...`) — these are intentional
  because the existing fixture style differs per file and is best read
  in place. Other tasks have complete code.

- **Type consistency:**
  - `EvalSnapshotPointer.kind` and `LoadedEvalSnapshot.kind` both use the
    same `str` form (`"split_view"` | `"external"`).
  - `evaluate(...)` always returns `EvaluateResult`.
  - `hash_key` is always a sorted `tuple[str, ...]` internally and
    a JSON list `list[str]` in manifests/manifests. User-facing call
    sites accept `str | list[str] | tuple[str, ...]` and normalize via
    `_normalize_hash_key`.
  - Predictions parquet columns are `(*hash_key, y_pred, y_proba_*)`
    consistently in `write_predictions_gcs` (Task 15) and consumers.
  - `EvalSpec.evaluate` consistently returns `{"primary": str, "metrics": [...]}`
    after Task 17; later tasks rely on this.
  - `LoadedEvalSnapshot.row_ids` is a `pd.DataFrame` of the hash_key
    projection (one column per hash_key element). The predictions writer
    accepts an `id_frame: pd.DataFrame` for the same reason.

---

## Execution Handoff

This is a 32-task plan spanning core library, runner integration, CLI,
notebooks, and contracts. Tasks 1–14 are independent foundation work
(data + eval snapshot identity, publish, load). Tasks 15–22 build the
predictions writer + EvalSpec redesign + evaluate verb. Tasks 23+ wire
the verb into the trial-time runner and clean up the surrounding paths.

Suggested execution: **Subagent-Driven** (one fresh subagent per task,
review between tasks) is the right tool for a plan this size — keeps each
task's context small and prevents drift across the 32 task boundaries.
