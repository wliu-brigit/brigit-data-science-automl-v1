"""Serializable split predicates: criteria are data, not code (design §12).

A split is a named, durable row-criterion over an immutable dataset. The
record form is a small JSON AST aligned with pyarrow's filter vocabulary;
``Where`` is a thin builder over it. No lambdas — trial contracts and eval
split-view identities must serialize and hash what a split *means*.

Authoring footgun: Python chains comparisons, so ``x == Where("c") < 80``
silently evaluates to ``(x == Where("c")) and (Where("c") < 80)`` — keep the
column on the left and parenthesize each clause when composing.
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


def _check_comparison_value(value: Any) -> Any:
    # Comparing against None silently matches nothing in pandas (even null
    # rows) — nullness has its own ops, so reject the trap at the edge.
    if value is None:
        raise TypeError(
            "comparison predicates cannot take None — use .is_null() / .not_null()"
        )
    return _check_scalar(value)


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
            return cls(op=op, column=column, value=_check_comparison_value(payload.get("value")))
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

    def _resolve_column(self, df: pd.DataFrame) -> pd.Series:
        if self.column in df.columns:
            return df[self.column]
        # Case-insensitive fallback: the data pipeline normalizes column names
        # while predicates are hand-written in configs — a pure case mismatch
        # (Where("EVAL_PCT") vs eval_pct) should resolve, not surface as a
        # KeyError minutes into a trial. Ambiguity still errors.
        wanted = str(self.column).casefold()
        matches = [name for name in df.columns if str(name).casefold() == wanted]
        if len(matches) == 1:
            return df[matches[0]]
        detail = f"is ambiguous between {matches}" if matches else "is missing"
        raise KeyError(
            f"split predicate references column {self.column!r}, which {detail}; "
            f"available: {sorted(df.columns)}"
        )

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
        series = self._resolve_column(df)
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
        # Known semantic gap vs mask(), to reconcile WHEN the push-down
        # reader is wired (it is deliberately unwired today): pyarrow
        # propagates nulls through comparisons, so `!=` and `~(==)` DROP
        # null rows where pandas keeps them; nullable pandas dtypes
        # (Int64/boolean) side with pyarrow. mask() is the authoritative
        # evaluator until parity is settled.
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
        return Predicate(op="<", column=self._column, value=_check_comparison_value(value))

    def __le__(self, value: Any) -> Predicate:
        return Predicate(op="<=", column=self._column, value=_check_comparison_value(value))

    def __gt__(self, value: Any) -> Predicate:
        return Predicate(op=">", column=self._column, value=_check_comparison_value(value))

    def __ge__(self, value: Any) -> Predicate:
        return Predicate(op=">=", column=self._column, value=_check_comparison_value(value))

    def __eq__(self, value: Any) -> Predicate:  # type: ignore[override]
        return Predicate(op="==", column=self._column, value=_check_comparison_value(value))

    def __ne__(self, value: Any) -> Predicate:  # type: ignore[override]
        return Predicate(op="!=", column=self._column, value=_check_comparison_value(value))

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
