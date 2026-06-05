"""Runner timing artifact publishing helpers."""

from __future__ import annotations

from automl.mlflow import trial as mlflow_trial
from automl.mlflow.trial.artifacts import runner as runner_artifacts
from automl.trial.timing_summary import build_runner_timing_summary


def log_timing(run_id: str, timing: dict[str, object]) -> None:
    summary = build_runner_timing_summary(timing)
    runner_artifacts.write_timing(run_id, summary)
    metrics = {"timing.total_seconds": float(summary.get("total_seconds") or 0.0)}
    phases = summary.get("phases")
    if isinstance(phases, dict):
        for name, value in phases.items():
            metrics[f"timing.{name}_seconds"] = float(value)
    runner_detail = (
        summary.get("phase_details", {}).get("runner", {})
        if isinstance(summary.get("phase_details"), dict)
        else {}
    )
    runner_phases = runner_detail.get("phases", {}) if isinstance(runner_detail, dict) else {}
    if isinstance(runner_phases, dict):
        for name, value in runner_phases.items():
            metrics[f"timing.{name}_seconds"] = float(value)
        if "fit" in runner_phases:
            metrics["time.fit_seconds"] = float(runner_phases["fit"])
        if "evaluation" in runner_phases:
            metrics["time.eval_seconds"] = float(runner_phases["evaluation"])
        if "validation" in runner_phases:
            metrics["time.validation_seconds"] = float(runner_phases["validation"])
    metrics["time.total_seconds"] = float(summary.get("total_seconds") or 0.0)
    mlflow_trial.log_metrics(run_id, metrics)


__all__ = ["log_timing"]
