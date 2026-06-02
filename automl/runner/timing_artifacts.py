"""Runner timing artifact publishing helpers."""

from __future__ import annotations

from automl.mlflow import trial as mlflow_trial
from automl.mlflow.trial.artifacts import runner as runner_artifacts


def log_timing(run_id: str, timing: dict[str, object]) -> None:
    runner_artifacts.write_timing(run_id, timing)
    metrics = {"timing.total_seconds": float(timing.get("total_seconds") or 0.0)}
    phases = timing.get("phases")
    if isinstance(phases, dict):
        for name, value in phases.items():
            metrics[f"timing.{name}_seconds"] = float(value)
        if "fit" in phases:
            metrics["time.fit_seconds"] = float(phases["fit"])
        if "evaluation" in phases:
            metrics["time.eval_seconds"] = float(phases["evaluation"])
        if "validation" in phases:
            metrics["time.validation_seconds"] = float(phases["validation"])
    metrics["time.total_seconds"] = float(timing.get("total_seconds") or 0.0)
    mlflow_trial.log_metrics(run_id, metrics)


__all__ = ["log_timing"]
