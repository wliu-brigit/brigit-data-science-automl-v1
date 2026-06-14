"""Beam-search subgroup core: planted conjunctions are recovered, rigor intact."""

import numpy as np
import pytest

from projects.fraud_anomaly_detection.analysis.subgroup_core import (
    beam_search,
    validate_rules,
)

pytestmark = pytest.mark.unit


@pytest.fixture()
def planted():
    rng = np.random.default_rng(7)
    n = 4000
    a = rng.random(n) < 0.3
    b = rng.random(n) < 0.3
    c = rng.random(n) < 0.3  # noise selector
    y = ((rng.random(n) < 0.02) | (a & b & (rng.random(n) < 0.8))).astype(int)
    return {"A": a, "B": b, "C": c}, y


def test_beam_search_recovers_planted_conjunction(planted):
    sels, y = planted
    rules, evaluated = beam_search(
        list(sels.items()), y, depth=2, beam_width=10, min_support=50)
    best = max(rules.items(), key=lambda kv: y[kv[1]].mean())
    assert set(best[0]) == {"A", "B"}
    assert evaluated > len(sels)  # depth-2 actually explored conjunctions


def test_beam_search_respects_min_support(planted):
    sels, y = planted
    rules, _ = beam_search(list(sels.items()), y, depth=3, beam_width=10,
                           min_support=50)
    assert all(mask.sum() >= 50 for mask in rules.values())


def test_validate_rules_scores_on_test_and_dedups(planted):
    sels, y = planted
    half = len(y) // 2
    tr = {k: v[:half] for k, v in sels.items()}
    te = {k: v[half:] for k, v in sels.items()}
    rules, _ = beam_search(list(tr.items()), y[:half], depth=2, beam_width=10,
                           min_support=50)
    rows = validate_rules(rules, te, y[half:], dpd_test=y[half:],
                          base_test=y[half:].mean(), min_test=30)
    assert rows, "expected at least one validated rule"
    top = rows[0]
    assert set(top) >= {"conds", "n_te", "never_te", "lift", "p"}
    assert top["never_te"] == max(r["never_te"] for r in rows)
    assert len({r["conds"] for r in rows}) == len(rows)


def test_validate_rules_keeps_distinct_footprints_with_same_stats():
    y_test = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    dpd_test = y_test.copy()
    selectors_test = {
        "A": np.array([True, True, True, True, False, False, False, False]),
        "B": np.array([False, False, False, False, True, True, True, True]),
    }
    train_mask = np.ones_like(y_test, dtype=bool)
    all_rules = {
        frozenset({"A"}): train_mask,
        frozenset({"B"}): train_mask,
    }

    rows = validate_rules(
        all_rules,
        selectors_test,
        y_test,
        dpd_test=dpd_test,
        base_test=float(y_test.mean()),
        min_test=1,
    )

    assert {row["conds"] for row in rows} == {"A", "B"}


def test_validate_rules_dedups_identical_footprints_by_mask():
    mask = np.array([True, True, True, True, False, False])
    y_test = np.array([1, 1, 0, 0, 0, 0])
    selectors_test = {
        "A": mask,
        "B": mask.copy(),
    }
    train_mask = np.ones_like(y_test, dtype=bool)
    all_rules = {
        frozenset({"A"}): train_mask,
        frozenset({"A", "B"}): train_mask,
    }

    rows = validate_rules(
        all_rules,
        selectors_test,
        y_test,
        dpd_test=y_test,
        base_test=float(y_test.mean()),
        min_test=1,
    )

    assert [row["conds"] for row in rows] == ["A"]
