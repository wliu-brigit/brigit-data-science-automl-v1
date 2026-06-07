"""Project-owned eval metrics for fraud_anomaly_detection.

These implement the automl.eval.Metric protocol and ride along in
EVAL.metrics; AveragePrecision stays the primary the loop optimizes. They are
instrumentation for the proxy-label phase: is_fraud is a threshold on the
upstream heuristic, so these report where the model and the heuristic agree,
disagree, and how the score relates to observed early-default outcomes. The
heuristic/outcome columns they read are eval-only — the feature registry
excludes them from features; reading them here is legitimate because metrics
run after scoring, never inside fit.

Depths are fractions of the scored population ("review the top k%"), mirroring
the industry detection-at-review-rate framing. Counts, not dollars, by design
for the baseline phase.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from automl.eval import Metric

from projects.fraud_anomaly_detection.scenarios import (
    SCENARIOS,
    SCENARIOS_VERSION,
    TRIGGER_COLUMNS,
    assign,
    residual_mask,
)

# Review depths: top 0.5% / 1% / 5% of scored rows.
DEFAULT_DEPTHS = (0.005, 0.01, 0.05)

BAND_ORDER = ("LOW", "POSSIBLE", "LIKELY", "EXTREMELY_LIKELY")


def _validated_depths(depths: Sequence[float]) -> tuple[float, ...]:
    if not depths:
        raise ValueError("at least one review depth is required")
    out = tuple(float(depth) for depth in depths)
    if any(not 0 < depth <= 1 for depth in out):
        raise ValueError(f"depths must be fractions in (0, 1], got {out}")
    return out


def _top_mask(scores: np.ndarray, depth: float) -> np.ndarray:
    """Boolean mask of the top-`depth` fraction of rows by score (desc)."""
    n = len(scores)
    k = max(1, int(round(n * depth)))
    order = np.argsort(-scores, kind="stable")
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


def _depth_records(y_true: np.ndarray, scores: np.ndarray, depths: Sequence[float]) -> list[dict[str, Any]]:
    total_pos = int(y_true.sum())
    records: list[dict[str, Any]] = []
    for depth in depths:
        mask = _top_mask(scores, depth)
        k = int(mask.sum())
        tp = int(y_true[mask].sum())
        records.append(
            {
                "depth": depth,
                "n_reviewed": k,
                "true_positives": tp,
                "precision": tp / k,
                "recall": (tp / total_pos) if total_pos else None,
                "fp_per_tp": ((k - tp) / tp) if tp else None,
            }
        )
    return records


class PrecisionRecallAtDepth(Metric):
    """precision / recall / FP:TP against the target in the top-k% of scores."""

    name = "precision_recall_at_depth"

    def __init__(self, *, depths: Sequence[float] = DEFAULT_DEPTHS) -> None:
        self.depths = _validated_depths(depths)

    def compute(self, df: pd.DataFrame, y_pred: Any, target_col: str) -> list[dict[str, Any]]:
        y_true = np.asarray(df[target_col], dtype=float)
        scores = np.asarray(y_pred, dtype=float)
        return _depth_records(y_true, scores, self.depths)


class BandReport(Metric):
    """Per heuristic band: count, mean score percentile, capture at each depth.

    The LOW row at each depth is the discovery queue: rows the heuristic
    called clean that the model ranks near the top. The proxy-label AP counts
    those as false positives; this is where they get counted as candidates.
    """

    name = "band_report"
    required_columns = ("heuristic_fraud_band",)

    def __init__(self, *, depths: Sequence[float] = DEFAULT_DEPTHS) -> None:
        self.depths = _validated_depths(depths)

    def compute(self, df: pd.DataFrame, y_pred: Any, target_col: str) -> list[dict[str, Any]]:
        del target_col  # band view is target-free by design
        bands = df["heuristic_fraud_band"].astype(str).to_numpy()
        scores = np.asarray(y_pred, dtype=float)
        # Percentile of each row's score within this eval frame (0-100,
        # higher = more anomalous): raw scores are not comparable across
        # model families, ranks are.
        percentile = pd.Series(scores).rank(method="average", pct=True).to_numpy() * 100
        top_masks = {depth: _top_mask(scores, depth) for depth in self.depths}
        seen = list(BAND_ORDER) + sorted(set(bands) - set(BAND_ORDER))
        records: list[dict[str, Any]] = []
        for band in seen:
            members = bands == band
            n = int(members.sum())
            record: dict[str, Any] = {
                "band": band,
                "n": n,
                "mean_score_percentile": float(percentile[members].mean()) if n else None,
            }
            for depth, mask in top_masks.items():
                in_top = int((members & mask).sum())
                record[f"capture_at_{depth}"] = (in_top / n) if n else None
                record[f"n_in_top_{depth}"] = in_top
            records.append(record)
        return records


class EarlyDefaultCapture(Metric):
    """precision / recall at depth against the gross-DPD45 outcome (rates only).

    The non-circular ruler: early default is an observed outcome, independent
    of the heuristic's feature thresholds. Restricted to rows old enough to
    judge (label_mature_d45 == 1). Early default includes innocent credit
    risk, so expect moderate capture — direction matters more than level.
    """

    name = "early_default_capture"
    required_columns = ("label_gross_dpd45", "label_mature_d45")

    def __init__(self, *, depths: Sequence[float] = DEFAULT_DEPTHS) -> None:
        self.depths = _validated_depths(depths)

    def compute(self, df: pd.DataFrame, y_pred: Any, target_col: str) -> dict[str, Any]:
        del target_col  # evaluated against the outcome label, not the task target
        mature = np.asarray(df["label_mature_d45"], dtype=float) == 1
        n_mature = int(mature.sum())
        if n_mature == 0:
            return {"n_mature": 0, "n_dpd45": 0, "records": []}
        y_true = np.asarray(df["label_gross_dpd45"], dtype=float)[mature]
        scores = np.asarray(y_pred, dtype=float)[mature]
        return {
            "n_mature": n_mature,
            "n_dpd45": int(y_true.sum()),
            "records": _depth_records(y_true, scores, self.depths),
        }


class NeverPaidAveragePrecision(Metric):
    """AP of the score against never-paid DPD45 on mature rows.

    The bust-out cut (gross DPD45 *and* not repaid as of snapshot) is the
    honest outcome ruler: once the scenario register absorbs the heuristic's
    top band, the proxy is_fraud label has ~no positives left in the residual
    and AP against it is degenerate. This metric replaces it as the primary.
    Never-paid still includes innocent credit risk, so treat it as a
    direction signal — don't over-tune to small deltas.
    """

    name = "never_paid_average_precision"
    required_columns = ("label_gross_dpd45", "label_repaid_current_snapshot", "label_mature_d45")

    def compute(self, df: pd.DataFrame, y_pred: Any, target_col: str) -> float:
        del target_col  # evaluated against the outcome, not the task target
        from sklearn.metrics import average_precision_score

        mature = np.asarray(df["label_mature_d45"], dtype=float) == 1
        y_true = (
            (np.asarray(df["label_gross_dpd45"], dtype=float) == 1)
            & (np.asarray(df["label_repaid_current_snapshot"], dtype=float) == 0)
        )[mature]
        if not y_true.any():
            return 0.0  # no mature never-paid rows: keep the primary finite
        return float(average_precision_score(y_true, np.asarray(y_pred, dtype=float)[mature]))


class ResidualOnly(Metric):
    """Delegate computed only on rows no scenario matched.

    Scenario-matched rows are rule-handled (see scenarios.py); wrapping a
    model-performance metric in ResidualOnly makes it arithmetically as if
    those rows were never in the test set — no count, no denominator, no
    ranking. The matched rows surface exclusively through ScenarioIdentified.
    """

    def __init__(self, inner: Metric) -> None:
        self.inner = inner
        self.name = f"residual_{inner.resolved_name()}"
        inner_required = tuple(getattr(inner, "required_columns", ()))
        self.required_columns = inner_required + tuple(
            col for col in TRIGGER_COLUMNS if col not in inner_required
        )

    def compute(self, df: pd.DataFrame, y_pred: Any, target_col: str) -> Any:
        mask = residual_mask(df).to_numpy()
        residual_df = df.loc[mask].reset_index(drop=True)
        residual_pred = np.asarray(y_pred, dtype=float)[mask]
        return self.inner.compute(residual_df, residual_pred, target_col)


class ScenarioIdentified(Metric):
    """Per-scenario rule outcomes: the only eval that sees matched rows.

    Reports counts and never-paid-DPD45 validation (gross DPD45 and not
    repaid as of snapshot, on mature rows — the bust-out cut) per scenario,
    plus the register version so any trial can be read knowing which
    register it ran under. Rule outcomes, not model performance: y_pred is
    deliberately ignored.
    """

    name = "scenario_identified"
    required_columns = TRIGGER_COLUMNS + (
        "label_gross_dpd45",
        "label_repaid_current_snapshot",
        "label_mature_d45",
    )

    def compute(self, df: pd.DataFrame, y_pred: Any, target_col: str) -> dict[str, Any]:
        del y_pred, target_col  # rule outcomes are model-free by design
        flags = assign(df)
        mature = np.asarray(df["label_mature_d45"], dtype=float) == 1
        never_paid = (
            (np.asarray(df["label_gross_dpd45"], dtype=float) == 1)
            & (np.asarray(df["label_repaid_current_snapshot"], dtype=float) == 0)
        )
        records: list[dict[str, Any]] = []
        for scenario in SCENARIOS:
            matched = flags[f"scenario_{scenario.name}"].to_numpy()
            matched_mature = matched & mature
            n_mature = int(matched_mature.sum())
            n_never_paid = int((matched_mature & never_paid).sum())
            records.append(
                {
                    "name": scenario.name,
                    "title": scenario.title,
                    "tier": scenario.tier,
                    "status": scenario.status,
                    "n": int(matched.sum()),
                    "n_mature": n_mature,
                    "n_never_paid": n_never_paid,
                    "never_paid_rate": (n_never_paid / n_mature) if n_mature else None,
                }
            )
        return {
            "scenarios_version": SCENARIOS_VERSION,
            "n_rows": int(len(df)),
            "n_residual": int(flags["scenario_any"].eq(False).sum()),
            "scenarios": records,
        }


__all__ = [
    "BandReport",
    "EarlyDefaultCapture",
    "NeverPaidAveragePrecision",
    "PrecisionRecallAtDepth",
    "ResidualOnly",
    "ScenarioIdentified",
    "DEFAULT_DEPTHS",
]
