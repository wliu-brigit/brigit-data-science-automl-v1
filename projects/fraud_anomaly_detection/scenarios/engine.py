"""Scenario engine — loads the YAML register and runs it. Register-agnostic.

This module owns *how* scenarios evaluate: the YAML schema, the condition
compiler, and the mask algebra. It never knows which scenarios exist — the
definitions live in register.yaml, and adding or changing a scenario never
requires touching this file.

The condition vocabulary deliberately mirrors the library's ``Where()``
predicates (ops: ``== != < <= > >= isin notin is_null not_null``), plus one
derived field for the register's only arithmetic need:

    - column: loan_amount          # plain column condition
      op: ">"
      value: 100
    - hours_between: [later_ts, earlier_ts]   # elapsed-hours condition
      op: "<="
      value: 24

Semantics, per the SCENARIOS.md rubric:

- **trigger** — a conjunction: ALL conditions must hold.
- **disqualifiers** — release conditions: a row is released if ANY holds.
- Nulls never match (NaN comparisons are False; ``is_null``/``not_null``
  test missingness explicitly).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# title (display, defaults to name) and typology (published-pattern anchor,
# filled at grounding time) are optional; theory is not — a scenario without
# a fraud-intent story is a credit-risk rule in disguise.
_REQUIRED_FIELDS = ("name", "tier", "status", "entry_date", "theory", "trigger")

_COMPARISON_OPS: dict[str, Callable[[pd.Series, Any], pd.Series]] = {
    "==": lambda s, v: s == v,
    "!=": lambda s, v: s != v,
    "<": lambda s, v: s < v,
    "<=": lambda s, v: s <= v,
    ">": lambda s, v: s > v,
    ">=": lambda s, v: s >= v,
    "isin": lambda s, v: s.isin(v),
    "notin": lambda s, v: ~s.isin(v),
}
_NULL_OPS: dict[str, Callable[[pd.Series], pd.Series]] = {
    "is_null": lambda s: s.isna(),
    "not_null": lambda s: s.notna(),
}


@dataclass(frozen=True)
class Scenario:
    """One named scenario, mirroring the SCENARIOS.md rubric."""

    name: str  # short slug; flag column is scenario_<name>
    title: str
    typology: str  # published-typology anchor (FATF/FinCEN framing)
    tier: str  # "block" | "mitigate" | "review"
    status: str  # "draft" | "signed_off"
    entry_date: str  # ISO date the scenario entered the register
    theory: str  # why a *fraudster* behaves this way
    trigger: Callable[[pd.DataFrame], pd.Series]  # compiled conjunctive mask
    # Compiled release mask (ANY disqualifier holds). None = none codified yet;
    # matched = trigger AND NOT disqualifiers.
    disqualifiers: Callable[[pd.DataFrame], pd.Series] | None = None

    def matches(self, df: pd.DataFrame) -> pd.Series:
        mask = self.trigger(df).fillna(False).astype(bool)
        if self.disqualifiers is not None:
            mask &= ~self.disqualifiers(df).fillna(False).astype(bool)
        return mask


@dataclass(frozen=True)
class Register:
    """A loaded register: the scenarios plus what the engine derived from them."""

    version: str
    scenarios: tuple[Scenario, ...]
    trigger_columns: tuple[str, ...]  # every column any condition reads


def hours_between(later: pd.Series, earlier: pd.Series) -> pd.Series:
    """Elapsed hours, NaN-propagating; coerces strings (load-path dependent)."""
    delta = pd.to_datetime(later) - pd.to_datetime(earlier)
    return delta / pd.Timedelta(hours=1)


def _condition_series(df: pd.DataFrame, spec: dict[str, Any], *, where: str) -> pd.Series:
    if "hours_between" in spec:
        pair = spec["hours_between"]
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"{where}: hours_between takes [later, earlier], got {pair!r}")
        return hours_between(df[pair[0]], df[pair[1]])
    if "column" in spec:
        return df[spec["column"]]
    raise ValueError(f"{where}: condition needs 'column' or 'hours_between': {spec!r}")


def _compile_condition(spec: dict[str, Any], *, where: str) -> Callable[[pd.DataFrame], pd.Series]:
    op = spec.get("op")
    if op in _NULL_OPS:
        if "value" in spec:
            raise ValueError(f"{where}: op {op!r} takes no value")
        return lambda df: _NULL_OPS[op](_condition_series(df, spec, where=where))
    if op in _COMPARISON_OPS:
        if "value" not in spec:
            raise ValueError(f"{where}: op {op!r} requires a value")
        return lambda df: _COMPARISON_OPS[op](_condition_series(df, spec, where=where), spec["value"])
    raise ValueError(f"{where}: unknown op {op!r} (expected one of {sorted(_COMPARISON_OPS | _NULL_OPS)})")


def _compile_conditions(
    specs: Any, *, mode: str, where: str
) -> Callable[[pd.DataFrame], pd.Series]:
    """Compile a condition list: 'all' (trigger) or 'any' (disqualifiers)."""
    if not isinstance(specs, list) or not specs:
        raise ValueError(f"{where}: empty trigger/disqualifiers — list at least one condition")
    compiled = [_compile_condition(spec, where=f"{where}[{i}]") for i, spec in enumerate(specs)]

    def mask(df: pd.DataFrame) -> pd.Series:
        masks = [fn(df).fillna(False).astype(bool) for fn in compiled]
        out = masks[0]
        for extra in masks[1:]:
            out = (out & extra) if mode == "all" else (out | extra)
        return out

    return mask


def _condition_columns(specs: Sequence[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    for spec in specs:
        referenced = spec["hours_between"] if "hours_between" in spec else [spec.get("column")]
        columns.extend(col for col in referenced if col)
    return columns


def load_register(path: Path | str) -> Register:
    """Load and compile the first YAML document of the register file.

    The file may carry a second, machine-owned document (validation stats,
    written by validation.py); the engine ignores it.
    """
    docs = list(yaml.safe_load_all(Path(path).read_text()))
    if not docs or not isinstance(docs[0], dict):
        raise ValueError(f"register file {path} has no register document")
    doc = docs[0]
    version = str(doc.get("version") or "")
    if not version:
        raise ValueError(f"register file {path} is missing 'version'")
    scenarios: list[Scenario] = []
    columns: list[str] = []
    for raw in doc.get("scenarios") or []:
        name = raw.get("name", "<unnamed>")
        # present-but-empty trigger lists fall through to the compiler's
        # clearer "empty trigger" error; this catches absent/blank fields
        missing = [field for field in _REQUIRED_FIELDS if raw.get(field) in (None, "")]
        if missing:
            raise ValueError(f"scenario {name!r}: missing required field(s) {missing}")
        where = f"scenario {name!r}"
        trigger_specs = raw["trigger"]
        disqualifier_specs = raw.get("disqualifiers") or []
        scenarios.append(
            Scenario(
                name=str(raw["name"]),
                title=str(raw.get("title") or raw["name"]),
                typology=str(raw.get("typology") or ""),
                tier=str(raw["tier"]),
                status=str(raw["status"]),
                entry_date=str(raw["entry_date"]),
                theory=str(raw["theory"]).strip(),
                trigger=_compile_conditions(trigger_specs, mode="all", where=f"{where} trigger"),
                disqualifiers=(
                    _compile_conditions(disqualifier_specs, mode="any", where=f"{where} disqualifiers")
                    if disqualifier_specs
                    else None
                ),
            )
        )
        columns.extend(_condition_columns(trigger_specs))
        columns.extend(_condition_columns(disqualifier_specs))
    deduped = tuple(dict.fromkeys(columns))
    return Register(version=version, scenarios=tuple(scenarios), trigger_columns=deduped)


def evaluate(df: pd.DataFrame, scenarios: Sequence[Scenario]) -> pd.DataFrame:
    """Boolean flag frame: one scenario_<name> column per scenario + scenario_any.

    Returns a new frame aligned to df's index; the input is not mutated.
    """
    flags = pd.DataFrame(index=df.index)
    for scenario in scenarios:
        flags[f"scenario_{scenario.name}"] = scenario.matches(df)
    flags["scenario_any"] = flags.any(axis=1) if len(scenarios) else pd.Series(False, index=df.index)
    return flags


def residual(df: pd.DataFrame, scenarios: Sequence[Scenario]) -> pd.Series:
    """True for rows no scenario matched — the population the model owns."""
    return ~evaluate(df, scenarios)["scenario_any"]


__all__ = ["Register", "Scenario", "evaluate", "hours_between", "load_register", "residual"]
