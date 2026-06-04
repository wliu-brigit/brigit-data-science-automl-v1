# Step 4 — Flexible splits: serializable predicates

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A split becomes a **named, durable row-criterion over an immutable
dataset**: the `Where` builder + a serializable predicate AST replace bucket
ranges (hard cut, no dual vocabulary); slice loading filters by predicate;
trial contracts and eval split-view identities carry the serialized AST;
`SPLIT_PCT` becomes an ordinary column (`Where("SPLIT_PCT") < 80`).

**Architecture:** `automl/project/predicates.py` is the one home for the
noun: `Where` (thin builder) emits `Predicate` nodes whose record form is a
small JSON AST aligned with pyarrow's native filter vocabulary — we never
invent a filter engine: in-memory evaluation is pandas operators, and
`to_pyarrow()` compiles to `pyarrow.dataset` expressions for the future
push-down reader (layout + exploiter deliberately ship together later,
design §12). Two green commits: (1) the additive predicate module, (2) the
hard cut.

**Tech stack:** pyarrow (`pyarrow.dataset` expressions; already a
dependency), pandas.

**Source of truth:** `../design.md` §12, §14 step 4. Absorbs
`docs/to-do/time-based-splitting.md` (its open questions are answered by
§12). Resolved in conversation 2026-06-04: lean on pyarrow's native
vocabulary; column names in predicates are the **persisted frame's names**
(normalized; `SPLIT_PCT` canonical) — a missing column fails loudly at load.

**Prereqs:** Steps 1–3 landed.

---

## The vocabulary

```python
from automl.project import Splits, Where

Splits(
    train = Where("application_date") < "2026-03-01",
    test  = (Where("application_date") >= "2026-03-01") & (Where("SPLIT_PCT") < 50),
)
```

- Ops: `== != < <= > >= .isin([...]) .notin([...]) .is_null() .not_null()`,
  composed with `& | ~`. That is the whole needed vocabulary (design §12).
- Record form (JSON AST, what trial contracts and eval identities hash):
  - leaf: `{"op": "<", "column": "SPLIT_PCT", "value": 80}`
  - composite: `{"op": "and", "items": [<ast>, <ast>]}`, `{"op": "or", ...}`,
    `{"op": "not", "items": [<ast>]}`
- **Record, don't police:** no overlap/disjointness validation anywhere —
  overlapping splits are legitimate methodology; `unique_key` makes overlap
  measurable when anyone wants to check.
- Values are JSON scalars (str/int/float/bool/None; lists for `in`); the
  user matches the column's dtype (ISO date strings compare against string
  columns; for datetime columns use what pandas/pyarrow compare naturally).

---

## PART 1 — the predicate module (additive, commit 1)

### Task 1: `automl/project/predicates.py` (TDD)

**Files:**
- Create: `tests/unit/project/test_predicates.py`
- Create: `automl/project/predicates.py`
- Modify: `automl/project/__init__.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Where builder + serializable predicate AST."""

import pandas as pd
import pytest

from automl.project.predicates import Predicate, Where

pytestmark = pytest.mark.unit


def _frame():
    return pd.DataFrame(
        {
            "SPLIT_PCT": [5, 50, 95, 20],
            "application_date": ["2026-01-01", "2026-04-01", "2026-02-15", None],
            "amount": [10.0, 20.0, 30.0, 40.0],
        }
    )


def test_comparison_ops_build_leaf_nodes():
    predicate = Where("SPLIT_PCT") < 80
    assert predicate.to_dict() == {"op": "<", "column": "SPLIT_PCT", "value": 80}


@pytest.mark.parametrize(
    "predicate, expected_rows",
    [
        (Where("SPLIT_PCT") < 80, [0, 1, 3]),
        (Where("SPLIT_PCT") >= 80, [2]),
        (Where("SPLIT_PCT") == 50, [1]),
        (Where("SPLIT_PCT") != 50, [0, 2, 3]),
        (Where("application_date") < "2026-03-01", [0, 2]),
        (Where("amount").isin([10.0, 40.0]), [0, 3]),
        (Where("amount").notin([10.0, 40.0]), [1, 2]),
        (Where("application_date").is_null(), [3]),
        (Where("application_date").not_null(), [0, 1, 2]),
        ((Where("SPLIT_PCT") < 80) & (Where("amount") > 15.0), [1, 3]),
        ((Where("SPLIT_PCT") >= 80) | (Where("amount") < 15.0), [0, 2]),
        (~(Where("SPLIT_PCT") < 80), [2]),
    ],
)
def test_mask_selects_expected_rows(predicate, expected_rows):
    df = _frame()
    assert list(df.index[predicate.mask(df)]) == expected_rows


def test_round_trip_through_the_record_form():
    predicate = (Where("application_date") >= "2026-03-01") & (Where("SPLIT_PCT") < 50)
    rebuilt = Predicate.from_dict(predicate.to_dict())
    assert rebuilt == predicate
    assert rebuilt.to_dict() == predicate.to_dict()


def test_missing_column_fails_loudly_at_evaluation():
    with pytest.raises(KeyError, match="no_such_column"):
        (Where("no_such_column") < 1).mask(_frame())


def test_columns_lists_every_referenced_column():
    predicate = (Where("a") < 1) & ((Where("b") == 2) | ~Where("c").is_null())
    assert predicate.columns() == frozenset({"a", "b", "c"})


def test_to_pyarrow_filters_a_table_identically():
    pyarrow = pytest.importorskip("pyarrow")
    df = _frame()
    predicate = (Where("SPLIT_PCT") < 80) & (Where("amount") > 15.0)
    table = pyarrow.Table.from_pandas(df)
    filtered = table.filter(predicate.to_pyarrow()).to_pandas()
    assert sorted(filtered["amount"].tolist()) == sorted(
        df[predicate.mask(df)]["amount"].tolist()
    )


def test_values_must_be_json_scalars():
    with pytest.raises(TypeError, match="JSON"):
        Where("a") < object()


def test_repr_reads_like_the_declaration():
    assert repr(Where("SPLIT_PCT") < 80) == 'Where("SPLIT_PCT") < 80'
    assert "&" in repr((Where("a") < 1) & (Where("b") == 2))
```

- [ ] **Step 2:** `uv run pytest tests/unit/project/test_predicates.py -v` —
FAIL (module missing).

- [ ] **Step 3: Implement `automl/project/predicates.py`**

```python
"""Serializable split predicates: criteria are data, not code (design §12).

A split is a named, durable row-criterion over an immutable dataset. The
record form is a small JSON AST aligned with pyarrow's filter vocabulary;
``Where`` is a thin builder over it. No lambdas — trial contracts and eval
split-view identities must serialize and hash what a split *means*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd


_COMPARISONS = frozenset({"==", "!=", "<", "<=", ">", ">="})
_MEMBERSHIP = frozenset({"in", "not-in"})
_NULLNESS = frozenset({"is-null", "not-null"})
_LOGICAL = frozenset({"and", "or", "not"})
_LEAF_OPS = _COMPARISONS | _MEMBERSHIP | _NULLNESS
_SCALAR_TYPES = (str, int, float, bool, type(None))


def _check_scalar(value: Any) -> Any:
    if isinstance(value, _SCALAR_TYPES):
        return value
    raise TypeError(f"predicate values must be JSON scalars, got {type(value).__name__}")


@dataclass(frozen=True)
class Predicate:
    op: str
    column: str | None = None
    value: Any = None
    items: tuple["Predicate", ...] = field(default=())

    # --- composition ----------------------------------------------------
    def __and__(self, other: "Predicate") -> "Predicate":
        return Predicate(op="and", items=(self, _require_predicate(other)))

    def __or__(self, other: "Predicate") -> "Predicate":
        return Predicate(op="or", items=(self, _require_predicate(other)))

    def __invert__(self) -> "Predicate":
        return Predicate(op="not", items=(self,))

    # --- record form ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        if self.op in _LOGICAL:
            return {"op": self.op, "items": [item.to_dict() for item in self.items]}
        payload: dict[str, Any] = {"op": self.op, "column": self.column}
        if self.op in _COMPARISONS:
            payload["value"] = self.value
        elif self.op in _MEMBERSHIP:
            payload["value"] = list(self.value)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Predicate":
        op = str(payload.get("op", ""))
        if op in _LOGICAL:
            items = tuple(cls.from_dict(item) for item in payload.get("items", ()))
            if op == "not" and len(items) != 1:
                raise ValueError("'not' takes exactly one item")
            if op in ("and", "or") and len(items) < 2:
                raise ValueError(f"{op!r} takes at least two items")
            return cls(op=op, items=items)
        if op not in _LEAF_OPS:
            raise ValueError(f"unknown predicate op {op!r}")
        column = payload.get("column")
        if not isinstance(column, str) or not column:
            raise ValueError(f"predicate op {op!r} requires a column name")
        if op in _COMPARISONS:
            return cls(op=op, column=column, value=_check_scalar(payload.get("value")))
        if op in _MEMBERSHIP:
            values = tuple(_check_scalar(item) for item in payload.get("value", ()))
            return cls(op=op, column=column, value=values)
        return cls(op=op, column=column)

    # --- introspection ----------------------------------------------------
    def columns(self) -> frozenset[str]:
        if self.op in _LOGICAL:
            out: frozenset[str] = frozenset()
            for item in self.items:
                out |= item.columns()
            return out
        return frozenset({self.column}) if self.column else frozenset()

    # --- evaluation: pandas mask (in-memory frames) ------------------------
    def mask(self, df: pd.DataFrame) -> pd.Series:
        if self.op == "and":
            out = self.items[0].mask(df)
            for item in self.items[1:]:
                out = out & item.mask(df)
            return out
        if self.op == "or":
            out = self.items[0].mask(df)
            for item in self.items[1:]:
                out = out | item.mask(df)
            return out
        if self.op == "not":
            return ~self.items[0].mask(df)
        if self.column not in df.columns:
            raise KeyError(
                f"split predicate references missing column {self.column!r}; "
                f"available: {sorted(df.columns)}"
            )
        series = df[self.column]
        if self.op == "==":
            return series == self.value
        if self.op == "!=":
            return series != self.value
        if self.op == "<":
            return series < self.value
        if self.op == "<=":
            return series <= self.value
        if self.op == ">":
            return series > self.value
        if self.op == ">=":
            return series >= self.value
        if self.op == "in":
            return series.isin(list(self.value))
        if self.op == "not-in":
            return ~series.isin(list(self.value))
        if self.op == "is-null":
            return series.isna()
        return series.notna()  # not-null

    # --- evaluation: pyarrow expression (push-down target) -----------------
    def to_pyarrow(self):
        import pyarrow.dataset as ds

        if self.op == "and":
            out = self.items[0].to_pyarrow()
            for item in self.items[1:]:
                out = out & item.to_pyarrow()
            return out
        if self.op == "or":
            out = self.items[0].to_pyarrow()
            for item in self.items[1:]:
                out = out | item.to_pyarrow()
            return out
        if self.op == "not":
            return ~self.items[0].to_pyarrow()
        column = ds.field(self.column)
        if self.op == "==":
            return column == self.value
        if self.op == "!=":
            return column != self.value
        if self.op == "<":
            return column < self.value
        if self.op == "<=":
            return column <= self.value
        if self.op == ">":
            return column > self.value
        if self.op == ">=":
            return column >= self.value
        if self.op == "in":
            return column.isin(list(self.value))
        if self.op == "not-in":
            return ~column.isin(list(self.value))
        if self.op == "is-null":
            return column.is_null()
        return ~column.is_null()  # not-null

    def __repr__(self) -> str:
        if self.op in _COMPARISONS:
            return f'Where("{self.column}") {self.op} {self.value!r}'
        if self.op == "in":
            return f'Where("{self.column}").isin({list(self.value)!r})'
        if self.op == "not-in":
            return f'Where("{self.column}").notin({list(self.value)!r})'
        if self.op == "is-null":
            return f'Where("{self.column}").is_null()'
        if self.op == "not-null":
            return f'Where("{self.column}").not_null()'
        if self.op == "not":
            return f"~({self.items[0]!r})"
        joiner = " & " if self.op == "and" else " | "
        return joiner.join(f"({item!r})" for item in self.items)


def _require_predicate(value: Any) -> "Predicate":
    if not isinstance(value, Predicate):
        raise TypeError(
            f"predicates compose with other predicates, got {type(value).__name__} "
            "(did you forget Where(...)?)"
        )
    return value


class Where:
    """Column proxy: comparison/membership/nullness methods emit Predicate nodes."""

    def __init__(self, column: str) -> None:
        if not isinstance(column, str) or not column.strip():
            raise ValueError("Where(column) requires a non-empty column name")
        self._column = column

    def __lt__(self, value: Any) -> Predicate:
        return Predicate(op="<", column=self._column, value=_check_scalar(value))

    def __le__(self, value: Any) -> Predicate:
        return Predicate(op="<=", column=self._column, value=_check_scalar(value))

    def __gt__(self, value: Any) -> Predicate:
        return Predicate(op=">", column=self._column, value=_check_scalar(value))

    def __ge__(self, value: Any) -> Predicate:
        return Predicate(op=">=", column=self._column, value=_check_scalar(value))

    def __eq__(self, value: Any) -> Predicate:  # type: ignore[override]
        return Predicate(op="==", column=self._column, value=_check_scalar(value))

    def __ne__(self, value: Any) -> Predicate:  # type: ignore[override]
        return Predicate(op="!=", column=self._column, value=_check_scalar(value))

    __hash__ = None  # equality builds predicates; Where is not hashable

    def isin(self, values) -> Predicate:
        return Predicate(
            op="in", column=self._column, value=tuple(_check_scalar(item) for item in values)
        )

    def notin(self, values) -> Predicate:
        return Predicate(
            op="not-in", column=self._column, value=tuple(_check_scalar(item) for item in values)
        )

    def is_null(self) -> Predicate:
        return Predicate(op="is-null", column=self._column)

    def not_null(self) -> Predicate:
        return Predicate(op="not-null", column=self._column)


__all__ = ["Predicate", "Where"]
```

Fix the repr test expectation against the implementation
(`'Where("SPLIT_PCT") < 80'`) once written — repr is for humans; the test
pins it loosely.

- [ ] **Step 4:** Export from `automl/project/__init__.py`:
`from automl.project.predicates import Predicate, Where` (+ `__all__`).

- [ ] **Step 5:** `uv run pytest tests/unit/project/test_predicates.py -v` — PASS.

- [ ] **Step 6: Commit**

```bash
git add automl/project/predicates.py automl/project/__init__.py tests/unit/project/test_predicates.py
git commit -m "Add Where builder + serializable predicate AST (design step 4, part 1)"
```

---

## PART 2 — the hard cut (commit 2)

### Task 2: `Splits` becomes predicates-only

**Files:**
- Modify: `automl/project/run_config.py`
- Tests: wherever Splits is pinned —
  `grep -rln "Splits(" tests automl projects` and update each (known:
  `tests/unit/eval/test_eval_thin_path.py`, project validation/run-config
  tests; review note: `test_cli_catalog.py` has no `Splits(` construction —
  trust the grep, not memory).

- [ ] **Step 1: Replace the `Splits` block** in `run_config.py` (delete
`DEFAULT_SPLIT_RANGES`, `_coerce_ranges`, `_bucket_set`,
`_validate_no_cross_name_overlap`):

```python
from automl.project.predicates import Predicate, Where

DEFAULT_SPLIT_PREDICATES = {
    "train": Where("SPLIT_PCT") < 80,
    "test": Where("SPLIT_PCT") >= 80,
}


@dataclass(frozen=True, init=False)
class Splits:
    """Named, durable row-criteria over an immutable dataset.

    Values are Predicate expressions (see automl.project.predicates).
    Overlap is deliberately not policed — the harness records exactly what
    each named split meant for any trial and enforces nothing about
    disjointness (design §12).
    """

    predicates: Mapping[str, Predicate]

    def __init__(
        self,
        predicates: Mapping[str, Predicate] | None = None,
        *,
        train: Predicate | None = None,
        test: Predicate | None = None,
        **named: Predicate,
    ) -> None:
        raw: dict[str, Predicate] = {}
        has_explicit = predicates is not None or train is not None or test is not None or bool(named)
        if predicates is not None:
            raw.update(dict(predicates))
        if train is not None:
            raw["train"] = train
        if test is not None:
            raw["test"] = test
        raw.update(named)
        if not raw:
            if has_explicit:
                raise ValueError("Splits must define at least one named predicate")
            raw.update(DEFAULT_SPLIT_PREDICATES)
        for name, value in raw.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"split name must be a non-empty string, got {name!r}")
            if not isinstance(value, Predicate):
                raise TypeError(
                    f"split {name!r} must be a Where(...) predicate, got "
                    f"{type(value).__name__} — bucket ranges were removed; "
                    f'use Where("SPLIT_PCT") < 80'
                )
        object.__setattr__(self, "predicates", dict(raw))

    def resolve(self, name: str) -> Predicate:
        try:
            return self.predicates[name]
        except KeyError as exc:
            known = ", ".join(sorted(self.predicates))
            raise KeyError(f"split {name!r} is not defined; known splits: {known}") from exc

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "Splits":
        raw = payload.get("predicates", payload)
        if not isinstance(raw, Mapping):
            raise ValueError("Splits payload must contain a 'predicates' mapping")
        return cls({str(name): Predicate.from_dict(ast) for name, ast in raw.items()})

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicates": {name: predicate.to_dict() for name, predicate in self.predicates.items()}
        }
```

`buckets()`/`train_buckets()`/`test_buckets()` are deleted; run
`grep -rn "train_buckets\|test_buckets\|\.buckets(" automl tests` and update
the callers (each becomes a `resolve(name)` + predicate use).

### Task 3: slice loading by predicate

**Files:**
- Modify: `automl/data/dataset.py` (`LoadedSlice`)
- Modify: `automl/data/registry.py`
- Modify: `automl/data/contract.py`
- Modify: `automl/runner/trial.py`

- [ ] **Step 1: `LoadedSlice`** — `split_ranges: tuple[tuple[int, int], ...]`
→ `predicate: Any` (a `Predicate`; typed `Any` to keep the data layer's
import surface unchanged — the object comes from `automl.project`, which
`registry.py` already imports).

- [ ] **Step 2: `registry.py`** — `load_dataset_by_id` /` load_dataset` /
`load_dataset_by_trial` replace `split_range` with `predicate`:

```python
def load_dataset_by_id(
    dataset_id: str,
    *,
    split_name: str | None = None,
    predicate: Predicate | None = None,
    session: Session | None = None,
) -> LoadedDataset | LoadedSlice:
    if split_name is not None and predicate is not None:
        raise ValueError("split_name and predicate are mutually exclusive")
    ...
    resolved = _resolve_predicate(active, split_name=split_name, predicate=predicate)
    if resolved is None:
        return loaded
    sliced = df[resolved.mask(df)].reset_index(drop=True)
    return LoadedSlice(
        dataset=dataset, df=sliced, registry=registry,
        split_name=split_name, predicate=resolved,
    )


def _resolve_predicate(active, *, split_name, predicate):
    if split_name is None and predicate is None:
        return None
    if split_name is not None:
        return active.config.require_run_config().splits.resolve(split_name)
    return predicate
```

`load_dataset_by_trial`: `contract.splits[split_name]` now holds an AST —
wrap with `Predicate.from_dict(...)`; the `split_range` normalization helpers
(`_normalize_split_range`, `_buckets`) are deleted. Bucket-range filtering
(`df[dataset.split_pct_col].isin(buckets)`) disappears with them — the
predicate mask is the only slicing path.

- [ ] **Step 3: `contract.py`** — `SliceContract.ranges` → `predicate`
(stored as the raw AST mapping):

```python
@dataclass(frozen=True)
class SliceContract:
    name: str | None
    predicate: Mapping[str, Any]
    n_rows: int
    content_hash: str
```

`to_dict`/`from_dict` carry the AST verbatim. `TrialDataContract.splits`
becomes `dict[str, Mapping[str, Any]]` (name → AST); its `to_dict`/`from_dict`
pass ASTs through unchanged.

- [ ] **Step 4: `runner/trial.py` `_trial_data_contract`** —

```python
    for name, predicate in run_config.splits.predicates.items():
        ...
        slices.append(
            SliceContract(
                name=name,
                predicate=loaded.predicate.to_dict(),
                n_rows=loaded.n_rows,
                content_hash=dataframe_content_hash(loaded.df),
            )
        )
    return TrialDataContract(
        ...,
        splits={name: predicate.to_dict() for name, predicate in run_config.splits.predicates.items()},
        slices=tuple(slices),
    )
```

- [ ] **Step 5:** Update the pinned tests:
- `tests/integration/data_pipeline/test_trial_replay.py` — `_write_trial_contract`
  (~144–181) builds `SliceContract(ranges=...)` / `splits={...: ((0,50),)}`
  → ASTs. **`test_load_dataset_by_id_accepts_disjoint_multi_range` (~185) is
  not a 1:1 rename** (review finding): it exercises ad-hoc disjoint slicing
  via the deleted `split_range=` kwarg. **Rewrite it** as the predicate
  equivalent — e.g.
  `predicate=(Where("SPLIT_PCT") >= 80) & (Where("SPLIT_PCT") < 90) | (Where("SPLIT_PCT") >= 95)`
  (or `.isin(...)` over the bucket list) asserting the same row selection and
  `sliced.predicate.to_dict()` round-trip — the *capability* (disjoint
  ad-hoc slices) must stay covered.
- `tests/unit/data/test_contract_validators.py` (SliceContract fixtures).
- `tests/integration/data_pipeline/test_materialize_load.py` (slice
  assertions compare against `predicate.mask`-selected rows).
- `tests/unit/eval/test_eval_thin_path.py:253` — the
  `fake_load_dataset_by_id(..., split_range=None, ...)` mock signature flips
  to `predicate=None` (it pins the kwarg `_load.py` now passes).

### Task 4: eval split views carry the predicate

**Files:**
- Modify: `automl/eval/eval_dataset.py`
- Modify: `automl/eval/prepare.py`
- Modify: `automl/eval/_load.py`
- Tests: `tests/unit/eval/test_eval_dataset_identity.py`,
  `tests/unit/eval/test_eval_thin_path.py`,
  `tests/integration/eval/*`

- [ ] **Step 1: `eval_dataset.py`** — `EvalDataset` drops `split_pct_col`
and `buckets`, gains `predicate: Mapping[str, Any] | None = None`. **Pinned
contract (review finding — the two halves must agree):**
`compute_eval_dataset_identity` takes the **AST mapping**
(`predicate: Mapping[str, Any]`); `split_view(...)` takes the **`Predicate`
object** and converts once at the boundary — it calls
`compute_eval_dataset_identity(..., predicate=predicate.to_dict(), ...)` and
stores the same AST on the field. Identity payload:

```python
        payload = {
            "schema_version": 1,
            "kind": kind,
            "of_dataset_id": of_dataset_id,
            "predicate": dict(predicate),   # predicate is already the AST mapping here
            "target_column": target_column,
            "unique_key": normalized_unique_key,
        }
```

`split_view(...)` signature: `(session, *, of_dataset_id, split, predicate,
target_column, unique_key)` — `predicate: Predicate`. `_normalize_buckets`
is deleted. `from_dict`/`to_dict` move the `predicate` AST instead of
`split_pct_col`/`buckets`. (`ev_` ids change; forward-only.)

- [ ] **Step 2: `prepare.py` `_prepare_split_view`** —

```python
    predicate = active.config.require_run_config().splits.resolve(split)
    parent = _dataset_by_id(data.list_datasets(session=active), dataset_id)
    recipe = EvalDataset.split_view(
        session=active,
        of_dataset_id=parent.id,
        split=split,
        predicate=predicate,
        target_column=parent.target_column,
        unique_key=parent.unique_key,
    )
```

- [ ] **Step 3: `_load.py`** split_view branch —

```python
        from automl.project.predicates import Predicate

        loaded = data.load_dataset_by_id(
            recipe.of_dataset_id,
            predicate=Predicate.from_dict(recipe.predicate),
            session=active,
        )
```

- [ ] **Step 4:** Update the eval tests: identity tests build predicates and
assert the recipe-based properties (same predicate → same `ev_` id;
different predicate → different id); persistence tests round-trip the AST
through `eval_dataset.json`.

### Task 5: configs, scaffold, CLI surface, docs

**Files:**
- Modify: `projects/example_homecredit/config.py`
- Modify: `projects/fraud_anomaly_detection/config.py`
- Modify: `automl/project/scaffold.py`
- Modify: `agent-skills/references/setup/run-config.md`
- Check: `automl/cli/**`, `automl/agent/proposer_context.py`,
  `automl/trial/**` for range survivors

- [ ] **Step 1: Configs** — in both project configs and the scaffold
template, the import line gains `Where` and:

```python
    splits=Splits(train=Where("SPLIT_PCT") < 80, test=Where("SPLIT_PCT") >= 80),
```

Scaffold comments show the time-based example from §12 (and that rolling/
backtesting windows are just a family of named splits). Update
`tests/unit/project/test_metadata_and_scaffold.py` if it pins template text.

- [ ] **Step 2: Survivor sweep** —

```bash
grep -rn "split_range\|split_ranges\|\[(0, 80)\]\|\[(80, 100)\]\|(0, 80)\|ranges=" automl tests projects agent-skills --include="*.py" --include="*.md" | grep -v docs/archive
```

Update every hit (CLI verbs exposing `--split-range` if any exist, proposer
context renderings of splits, trial templates). The range API must not
survive anywhere — one split vocabulary, not two.

- [ ] **Step 3: Docs** — `agent-skills/references/setup/run-config.md`:
rewrite the Splits section around `Where`, the op set, record-don't-police,
column-availability-is-the-only-requirement, and `SPLIT_PCT` as an ordinary
column. Check `tests/contracts/test_data_docs_truth.py` (pins "requires
named splits" phrasing — named splits still exist, so likely unaffected;
update if it pins range syntax).

- [ ] **Step 4: Archive the absorbed to-do** — per the docs lifecycle
(`docs/README.md`): `git mv docs/to-do/time-based-splitting.md
docs/archive/2026-06-time-based-splitting.md` (prefix per the archive's
naming convention — check `ls docs/archive/` and match), and remove the
HANDOFF/README links pointing at it.

### Task 6: green suite, handoff, commit 2

- [ ] **Step 1:** `uv run pytest tests/unit tests/contracts tests/integration` — PASS.

- [ ] **Step 2:** Update `docs/HANDOFF.md`: all four steps of the effort
landed; tail-end activities remain (live notebook verification on
example_homecredit — note
`projects/example_homecredit/notebooks/2_run_agent_automl.ipynb` contains a
`Splits(train=[(0, 80)]...)` cell that now raises and must be updated in
that pass; fraud project first real materialize).

- [ ] **Step 3: Commit**

```bash
git add -A automl tests projects agent-skills docs
git commit -m "Hard-cut Splits to Where(...) predicates; contracts and eval identities carry the AST

Slice loading filters by predicate; bucket ranges removed everywhere;
SPLIT_PCT is an ordinary column (design step 4)."
```

---

## Self-review checklist

- [ ] No lambda/callable path anywhere in splits — criteria serialize as
  JSON ASTs end to end (declare → trial contract → eval identity → replay).
- [ ] `load_dataset_by_trial` replays a historical trial's split from the
  serialized AST alone (test_trial_replay proves it).
- [ ] Overlapping splits construct and record without complaint.
- [ ] A predicate naming a missing column fails at load with the column
  name and the available columns.
- [ ] Push-down/`to_pyarrow` is implemented but **not wired** into the
  reader — sorting the persisted frame ships with the push-down reader
  later, as one unit (design §12). No premature `filters=` in
  `read_parquet`.
- [ ] `grep -rn "split_range\|train_buckets\|_coerce_ranges" automl tests` → empty.
- [ ] Default `Splits()` still yields train/test names with 80/20 semantics.
