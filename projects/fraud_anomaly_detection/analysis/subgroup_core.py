"""Reusable subgroup-discovery core: beam search + held-out validation.

Lifted verbatim from subgroup_discovery.py (2026-06-10) so the same proven
machinery can sweep any feature frame — the pinned-dataset lens and the
graph-feature sweep both consume THIS. The rigor rules live here: minimum
train support, held-out test validation, candidate counting for the
Bonferroni eyeball, identical-footprint dedup keeping the shortest rule.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def binom_sf(k: int, n: int, p: float) -> float:
    """One-sided P(X >= k) under Binomial(n, p), normal approximation with a
    continuity correction — significance of the test precision vs base."""
    from math import erfc, sqrt
    if n == 0 or p <= 0 or p >= 1:
        return 1.0
    z = (k - 0.5 - n * p) / sqrt(n * p * (1 - p))
    return 0.5 * erfc(z / sqrt(2))


def beam_search(
    selectors: list[tuple[str, np.ndarray]],
    y: np.ndarray,
    *,
    depth: int = 3,
    beam_width: int = 80,
    min_support: int = 80,
) -> tuple[dict[frozenset, np.ndarray], int]:
    """Beam-search conjunctions of selectors on TRAIN; quality = precision.

    Returns (all_rules: conjunction -> train mask, candidates_evaluated).
    Every returned rule meets the min_support floor.
    """
    def precision(mask: np.ndarray) -> float:
        n = int(mask.sum())
        return float(y[mask].mean()) if n else 0.0

    evaluated = 0
    beam: list[tuple[tuple[str, ...], np.ndarray]] = []
    for name, mask in selectors:
        evaluated += 1
        if int(mask.sum()) >= min_support:
            beam.append(((name,), mask))
    beam.sort(key=lambda b: precision(b[1]), reverse=True)
    beam = beam[:beam_width]

    seen: set[frozenset] = set(frozenset(c) for c, _ in beam)
    all_rules: dict[frozenset, np.ndarray] = {frozenset(c): m for c, m in beam}

    for _ in range(depth - 1):
        cand: list[tuple[tuple[str, ...], np.ndarray]] = []
        for conds, mask in beam:
            for name, sel_mask in selectors:
                if name in conds:
                    continue
                key = frozenset(conds + (name,))
                if key in seen:
                    continue
                new_mask = mask & sel_mask
                evaluated += 1
                if int(new_mask.sum()) < min_support:
                    continue
                seen.add(key)
                cand.append((tuple(sorted(key)), new_mask))
                all_rules[key] = new_mask
        if not cand:
            break
        cand.sort(key=lambda b: precision(b[1]), reverse=True)
        beam = cand[:beam_width]

    return all_rules, evaluated


def validate_rules(
    all_rules: dict[frozenset, np.ndarray],
    selectors_test: dict[str, np.ndarray],
    y_test: np.ndarray,
    *,
    dpd_test: np.ndarray,
    base_test: float,
    y_train: np.ndarray | None = None,
    min_test: int = 30,
) -> list[dict[str, Any]]:
    """Score every discovered rule on held-out TEST; dedup identical footprints.

    Returns rows sorted by test precision, each with conds / n_te / never_te /
    lift / never_tr (train precision, NaN when y_train omitted) / dpd_te / p
    (one-sided binomial vs base).
    """
    n_test = len(y_test)
    rows: list[dict[str, Any]] = []
    for key, train_mask in all_rules.items():
        mask_te = np.ones(n_test, bool)
        for name in key:
            mask_te = mask_te & selectors_test[name]
        n_te = int(mask_te.sum())
        if n_te < min_test:
            continue
        prec_te = float(y_test[mask_te].mean())
        never_tr = float("nan")
        if y_train is not None and train_mask.sum():
            never_tr = float(y_train[train_mask].mean())
        rows.append({
            "conds": " AND ".join(sorted(key)),
            "n_te": n_te,
            "never_te": prec_te,
            "lift": (prec_te / base_test) if base_test else float("nan"),
            "never_tr": never_tr,
            "dpd_te": float(dpd_test[mask_te].mean()),
            "p": binom_sf(int(round(prec_te * n_te)), n_te, base_test),
            "_mask_sig": np.packbits(mask_te).tobytes(),
        })
    # Dedup: collapse subgroups with an identical test footprint, keeping the
    # SHORTEST conjunction. Equal aggregate stats are not enough to dedup.
    rows.sort(key=lambda r: (r["_mask_sig"], r["conds"].count(" AND ")))
    deduped: dict[bytes, dict[str, Any]] = {}
    for row in rows:
        sig = row["_mask_sig"]
        if (sig not in deduped
                or row["conds"].count(" AND ") < deduped[sig]["conds"].count(" AND ")):
            deduped[sig] = row
    out = sorted(deduped.values(), key=lambda r: r["never_te"], reverse=True)
    for row in out:
        row.pop("_mask_sig", None)
    return out
