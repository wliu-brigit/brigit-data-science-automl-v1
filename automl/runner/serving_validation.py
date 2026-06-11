"""Serving validation artifact publishing helpers."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import automl.data as data
from automl.errors import ValidationError
from automl.mlflow import client as mlflow_client
from automl.mlflow import tags as mlflow_tags
from automl.mlflow import trial as mlflow_trial
from automl.mlflow.trial.artifacts import runner as runner_artifacts
from automl.project import Session
from automl.runner.timing import TimingRecorder, timed_phase


# Fallback wall-clock budget for the serving-validation subprocess when the
# session has no RUN_CONFIG. Projects tune this via
# RUN_CONFIG.serving_validation_seconds (default 300 — the observed full-data
# baseline sat at 120.07s, so the old 120s cap was boundary-tight).
_DEFAULT_VALIDATION_TIMEOUT_S = 300


def _validation_timeout_seconds(active: Session) -> int:
    run_config = getattr(active.config, "run_config", None)
    value = getattr(run_config, "serving_validation_seconds", None)
    return int(value) if value else _DEFAULT_VALIDATION_TIMEOUT_S


def _decode_tail(raw: object, *, limit: int = 1000) -> str:
    """Last ``limit`` chars of subprocess output, always as ``str``.

    ``TimeoutExpired.stderr`` is bytes even under ``text=True``.
    """
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw[-limit:].decode("utf-8", "replace")
    return str(raw)[-limit:]


def log_validation_artifacts(
    *,
    run_id: str,
    active: Session,
    model,
    dataset_id: str,
    eval_split: str,
    model_registry,
    timing: TimingRecorder | None = None,
    model_uri: str | None = None,
    context=None,
) -> dict[str, object]:
    tolerance = 1e-10
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        input_csv = root / "input.csv"
        input_parquet = root / "input.parquet"
        expected_parquet = root / "expected.parquet"
        input_schema = root / "input_schema.json"
        latency_path = root / "latency_detail.json"
        report_path = root / "report.json"
        with timed_phase(timing, "validation_fixture"):
            loaded = data.load_dataset_by_id(dataset_id, split_name=eval_split, session=active)
            input_frame = _validation_input_frame(model_registry, loaded.df)
            if input_frame.empty:
                raise ValidationError("validation requires at least one eval row")
            input_frame = input_frame.head(min(10, len(input_frame))).reset_index(drop=True)
            expected_scores = _series_values(_predict_model(model, input_frame))
            if len(expected_scores) != len(input_frame):
                raise ValidationError(
                    "validation expected-score count does not match input rows"
                )

            import pandas as pd

            expected = pd.DataFrame(
                {
                    "row_id": list(range(len(input_frame))),
                    "expected_score": expected_scores,
                }
            )
            input_frame.to_csv(input_csv, index=False)
            input_frame.to_parquet(input_parquet, index=False)
            expected.to_parquet(expected_parquet, index=False)
            input_schema.write_text(
                json.dumps(
                    _input_schema_from_frame(model_registry, input_frame),
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
        # Publish the validation fixtures up front, before the (fallible)
        # benchmark subprocess runs. A failed or timed-out validation then still
        # leaves the input rows + expected scores in MLflow for debugging,
        # instead of discarding them with the temp dir.
        with timed_phase(timing, "validation_fixture_publish"):
            runner_artifacts.write_local_file(
                run_id,
                "validation/data/input.csv",
                input_csv,
            )
            runner_artifacts.write_local_file(
                run_id,
                "validation/data/input.parquet",
                input_parquet,
            )
            runner_artifacts.write_local_file(
                run_id,
                "validation/data/expected.parquet",
                expected_parquet,
            )
            runner_artifacts.write_local_file(
                run_id,
                "validation/data/input_schema.json",
                input_schema,
            )
        with timed_phase(timing, "validation"):
            raw_report = _run_pyfunc_validation(
                run_id=run_id,
                active=active,
                input_parquet=input_parquet,
                input_csv=input_csv,
                expected_parquet=expected_parquet,
                input_schema=input_schema,
                report_path=report_path,
                tolerance=tolerance,
                model_uri=model_uri,
                context=context,
            )
        with timed_phase(timing, "validation_publish"):
            if latency_path.exists():
                latency_detail = json.loads(latency_path.read_text(encoding="utf-8"))
            else:
                latency_detail = {
                    "schema_version": 1,
                    "status": "not_measured",
                    "reason": "validation_failed",
                }
                latency_path.write_text(
                    json.dumps(latency_detail, indent=2),
                    encoding="utf-8",
                )
                if context is not None:
                    context.record_issue(
                        "validation latency not measured (validation failed)",
                        phase="validation_publish",
                        severity="warning",
                    )
            report = _validation_report_document(
                raw_report,
                run_id=run_id,
                tolerance=tolerance,
            )
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            runner_artifacts.write_local_file(
                run_id,
                "validation/latency_detail.json",
                latency_path,
            )
            runner_artifacts.write_local_file(
                run_id,
                "validation/report.json",
                report_path,
            )
            _log_validation_tags_and_metrics(run_id, report)
            return report


def _validation_input_frame(registry, df):
    if registry is not None and hasattr(registry, "select"):
        return registry.select(df)
    return df


def _predict_model(model, frame):
    try:
        return model.predict(context=None, model_input=frame)
    except TypeError:
        return model.predict(frame)


def _series_values(values) -> list[float]:
    import pandas as pd

    if isinstance(values, pd.Series):
        return [float(value) for value in values.tolist()]
    raw = values.tolist() if hasattr(values, "tolist") else list(values)
    return [float(value) for value in raw]


def _input_schema_from_frame(registry, frame) -> dict[str, object]:
    registry_dtypes: dict[str, str] = {}
    if registry is not None and hasattr(registry, "to_dataframe"):
        try:
            registry_frame = registry.to_dataframe()
            registry_dtypes = {
                str(row["name"]): str(row.get("dtype", ""))
                for row in registry_frame.to_dict("records")
            }
        except Exception:
            registry_dtypes = {}
    return {
        "schema_version": 1,
        "features": [
            {
                "name": str(column),
                "dtype": registry_dtypes.get(str(column), str(dtype)),
            }
            for column, dtype in frame.dtypes.items()
        ],
    }


def _run_pyfunc_validation(
    *,
    run_id: str,
    active: Session,
    input_parquet: Path,
    input_csv: Path,
    expected_parquet: Path,
    input_schema: Path,
    report_path: Path,
    tolerance: float,
    model_uri: str | None = None,
    context=None,
) -> dict[str, object]:
    # Prefer the MLflow 3 logged-model URI (``models:/<id>``). It resolves
    # straight to the model's artifact location; the legacy ``runs:/<run>/model``
    # fallback first probes the (empty) run artifact path, which 500s and is then
    # retried.
    timeout_s = _validation_timeout_seconds(active)
    resolved_model_uri = model_uri or f"runs:/{run_id}/model"
    script = r"""
import json
import statistics
import sys
import time
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd

model_uri = sys.argv[1]
tracking_uri = sys.argv[2]
input_path = Path(sys.argv[3])
csv_input_path = Path(sys.argv[4])
expected_path = Path(sys.argv[5])
input_schema_path = Path(sys.argv[6])
result_path = Path(sys.argv[7])
tolerance = float(sys.argv[8])

WARMUP_ITERATIONS = 10
MEASURED_ITERATIONS = 100
REPEAT_GROUPS = 3
TRIM_FRACTION = 0.10


def _latency_percentile(values, percentile):
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    index = (len(values) - 1) * (percentile / 100.0)
    lower = int(np.floor(index))
    upper = int(np.ceil(index))
    if lower == upper:
        return float(values[lower])
    fraction = index - lower
    return float(values[lower] + ((values[upper] - values[lower]) * fraction))


def _latency_summary(samples_ms):
    if not samples_ms:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0,
        }
    sorted_samples = sorted(samples_ms)
    return {
        "mean": float(statistics.fmean(samples_ms)),
        "median": float(statistics.median(samples_ms)),
        "p95": _latency_percentile(sorted_samples, 95),
        "p99": _latency_percentile(sorted_samples, 99),
        "min": float(min(samples_ms)),
        "max": float(max(samples_ms)),
        "std": float(statistics.pstdev(samples_ms)) if len(samples_ms) > 1 else 0.0,
    }


def _trimmed_samples(samples_ms):
    sorted_samples = sorted(samples_ms)
    trim_n = int(len(sorted_samples) * TRIM_FRACTION)
    if trim_n and len(sorted_samples) > (2 * trim_n):
        return sorted_samples[trim_n:-trim_n], trim_n
    return sorted_samples, 0


def _latency_block(samples_ms):
    trimmed_samples, trim_n = _trimmed_samples(samples_ms)
    return {
        "sample_count": len(samples_ms),
        "trimmed_sample_count": len(trimmed_samples),
        "dropped_low_count": trim_n,
        "dropped_high_count": trim_n,
        "raw_latency_ms": _latency_summary(samples_ms),
        "latency_ms": _latency_summary(trimmed_samples),
    }


def _benchmark_single_row_latency(model, input_df, model_load_ms):
    if input_df.empty:
        raise RuntimeError("validation latency requires at least one input row")

    request = input_df.iloc[[0]].copy()
    groups = []
    all_samples_ms = []
    for group_index in range(REPEAT_GROUPS):
        for _ in range(WARMUP_ITERATIONS):
            np.asarray(model.predict(request))

        samples_ms = []
        for _ in range(MEASURED_ITERATIONS):
            started = time.perf_counter_ns()
            np.asarray(model.predict(request))
            samples_ms.append((time.perf_counter_ns() - started) / 1_000_000)
        all_samples_ms.extend(samples_ms)
        groups.append({"group": group_index + 1, **_latency_block(samples_ms)})

    aggregate = _latency_block(all_samples_ms)
    trim_policy = {
        "drop_each_tail_fraction": TRIM_FRACTION,
        "drop_each_tail_count": int(MEASURED_ITERATIONS * TRIM_FRACTION),
    }

    compact = {
        "method": "fresh_process_loaded_pyfunc_single_row_repeated",
        "model_load_ms": float(model_load_ms),
        "input_format": "csv_string",
        "repeat_groups": REPEAT_GROUPS,
        "warmup_iterations": WARMUP_ITERATIONS,
        "measured_iterations_per_group": MEASURED_ITERATIONS,
        "trim_fraction": TRIM_FRACTION,
        "request_row_count": 1,
        "sample_count": aggregate["sample_count"],
        "trimmed_sample_count": aggregate["trimmed_sample_count"],
        "raw_latency_ms": aggregate["raw_latency_ms"],
        "latency_ms": aggregate["latency_ms"],
    }
    details = {
        "schema_version": 1,
        "method": compact["method"],
        "model_load_ms": compact["model_load_ms"],
        "input_format": compact["input_format"],
        "repeat_groups": REPEAT_GROUPS,
        "warmup_iterations": WARMUP_ITERATIONS,
        "measured_iterations_per_group": MEASURED_ITERATIONS,
        "request_row_count": 1,
        "trim_policy": trim_policy,
        "sample_count": aggregate["sample_count"],
        "trimmed_sample_count": aggregate["trimmed_sample_count"],
        "groups": groups,
        "aggregate": aggregate,
    }
    return compact, details


def _input_schema_feature_names(input_schema):
    features = input_schema.get("features", [])
    if not isinstance(features, list):
        raise RuntimeError("input_schema.features must be a list")
    names = []
    for feature in features:
        if not isinstance(feature, dict) or not feature.get("name"):
            raise RuntimeError("input_schema.features entries must include name")
        names.append(str(feature["name"]))
    return names


def _validate_input_schema_columns(input_df, expected_columns):
    actual_columns = [str(column) for column in input_df.columns]
    missing = [column for column in expected_columns if column not in actual_columns]
    extra = [column for column in actual_columns if column not in expected_columns]
    return {
        "status": "passed" if not missing and not extra else "failed",
        "expected_column_count": len(expected_columns),
        "actual_column_count": len(actual_columns),
        "missing_columns": missing,
        "extra_columns": extra,
    }


def _all_values_are_strings(input_df):
    for column in input_df.columns:
        if not input_df[column].map(lambda value: isinstance(value, str)).all():
            return False
    return True


def _validate_predictions(model, input_df, expected, *, input_reader, input_dtype_mode):
    preds = np.asarray(model.predict(input_df), dtype=float).reshape(-1)
    if len(preds) != len(expected):
        raise RuntimeError(
            f"prediction count {len(preds)} does not match expected count {len(expected)}"
        )
    max_abs_diff = float(np.max(np.abs(preds - expected))) if len(expected) else 0.0
    return {
        "status": "passed" if max_abs_diff <= tolerance else "failed",
        "row_count": int(len(expected)),
        "max_abs_diff": max_abs_diff,
        "tolerance": tolerance,
        "input_reader": input_reader,
        "input_dtype_mode": input_dtype_mode,
    }


result_path.parent.mkdir(parents=True, exist_ok=True)
latency_details_path = result_path.parent / "latency_detail.json"
try:
    mlflow.set_tracking_uri(tracking_uri)
    load_started = time.perf_counter_ns()
    model = mlflow.pyfunc.load_model(model_uri)
    model_load_ms = (time.perf_counter_ns() - load_started) / 1_000_000
    input_df = pd.read_parquet(input_path)
    csv_input_df = pd.read_csv(csv_input_path, dtype=str, keep_default_na=False)
    expected_df = pd.read_parquet(expected_path).sort_values("row_id").reset_index(drop=True)
    input_schema = json.loads(input_schema_path.read_text())
    expected_columns = _input_schema_feature_names(input_schema)
    expected = expected_df["expected_score"].astype(float).to_numpy()
    input_schema_check = _validate_input_schema_columns(input_df, expected_columns)
    csv_input_schema_check = _validate_input_schema_columns(csv_input_df, expected_columns)
    if csv_input_schema_check["status"] != "passed":
        input_schema_check = {
            **input_schema_check,
            "status": "failed",
            "csv_missing_columns": csv_input_schema_check["missing_columns"],
            "csv_extra_columns": csv_input_schema_check["extra_columns"],
        }
    checks = {
        "input_schema": input_schema_check,
        "parquet_roundtrip": _validate_predictions(
            model,
            input_df,
            expected,
            input_reader="pandas.read_parquet",
            input_dtype_mode="native_parquet",
        ),
        "csv_string_roundtrip": {
            **_validate_predictions(
                model,
                csv_input_df,
                expected,
                input_reader="pandas.read_csv(dtype=str, keep_default_na=False)",
                input_dtype_mode="all_string",
            ),
            "all_values_string": _all_values_are_strings(csv_input_df),
        },
    }
    if not checks["csv_string_roundtrip"]["all_values_string"]:
        checks["csv_string_roundtrip"]["status"] = "failed"
    max_abs_diff = max(
        float(check["max_abs_diff"])
        for key, check in checks.items()
        if key.endswith("_roundtrip")
    )
    failed_checks = [
        name for name, check in checks.items()
        if isinstance(check, dict) and check.get("status") != "passed"
    ]
    if failed_checks:
        report = {
            "schema_version": 1,
            "status": "failed",
            "model_uri": model_uri,
            "row_count": int(len(expected)),
            "max_abs_diff": max_abs_diff,
            "tolerance": tolerance,
            "checks": checks,
            "failed_checks": failed_checks,
        }
        result_path.write_text(json.dumps(report, indent=2))
        raise SystemExit(1)

    latency, latency_details = _benchmark_single_row_latency(
        model,
        csv_input_df,
        model_load_ms,
    )
    latency_details_path.write_text(json.dumps(latency_details, indent=2))
    report = {
        "schema_version": 1,
        "status": "passed",
        "model_uri": model_uri,
        "row_count": int(len(expected)),
        "max_abs_diff": max_abs_diff,
        "tolerance": tolerance,
        "checks": checks,
        "latency": latency,
    }
    result_path.write_text(json.dumps(report, indent=2))
    raise SystemExit(0)
except Exception as exc:
    result_path.write_text(json.dumps({
        "schema_version": 1,
        "status": "failed",
        "model_uri": model_uri,
        "row_count": 0,
        "max_abs_diff": None,
        "tolerance": tolerance,
        "error": repr(exc),
    }, indent=2))
    raise
"""
    child_env = dict(os.environ)
    child_env["MLFLOW_TRACKING_URI"] = active.config.mlflow_tracking_uri
    child_env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Bound the child's retry backoff too: the parent env normally carries the
    # seam-wide cap (set at automl.mlflow.client import), but validation may
    # also be invoked standalone. ``setdefault`` respects an operator override.
    child_env.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", mlflow_client.HTTP_MAX_RETRIES)
    existing_pythonpath = child_env.get("PYTHONPATH", "")
    pythonpath_parts = [str(active.config.repo_root)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    child_env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                resolved_model_uri,
                active.config.mlflow_tracking_uri,
                str(input_parquet),
                str(input_csv),
                str(expected_parquet),
                str(input_schema),
                str(report_path),
                str(tolerance),
            ],
            cwd=str(active.config.repo_root),
            env=child_env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        # Don't crash the trial: record a specific failure report so the run has
        # something actionable (alongside the fixtures already published).
        stderr_tail = _decode_tail(exc.stderr)
        report = {
            "schema_version": 1,
            "status": "failed",
            "model_uri": resolved_model_uri,
            "row_count": 0,
            "max_abs_diff": None,
            "tolerance": tolerance,
            "error": (
                f"validation subprocess exceeded {timeout_s}s "
                f"loading/benchmarking {resolved_model_uri!r} "
                "(RUN_CONFIG.serving_validation_seconds)"
            ),
            "error_class": "TimeoutExpired",
            "stderr_tail": stderr_tail,
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if context is not None:
            context.record_issue(report["error"], phase="validation", severity="error")
        return report
    if completed.returncode < 0:
        signum = -completed.returncode
        try:
            signal_name = signal.Signals(signum).name
        except ValueError:
            signal_name = f"signal {signum}"
        report = {
            "schema_version": 1,
            "status": "failed",
            "model_uri": resolved_model_uri,
            "row_count": 0,
            "max_abs_diff": None,
            "tolerance": tolerance,
            "error": (
                f"validation subprocess died on {signal_name} "
                f"(returncode {completed.returncode}); any partial report is untrusted"
            ),
            "error_class": "SignalExit",
            "signal": signum,
            "signal_name": signal_name,
            "stderr_tail": _decode_tail(completed.stderr),
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if context is not None:
            context.record_issue(report["error"], phase="validation", severity="error")
        return report
    if report_path.exists():
        with report_path.open(encoding="utf-8") as handle:
            report = json.load(handle)
    else:
        report = {
            "schema_version": 1,
            "status": "failed",
            "model_uri": resolved_model_uri,
            "row_count": 0,
            "max_abs_diff": None,
            "tolerance": tolerance,
            "error": _decode_tail(completed.stderr),
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not isinstance(report, dict):
        raise ValidationError("validation report must be a JSON object")
    if completed.returncode != 0 and "error" not in report:
        report["error"] = _decode_tail(completed.stderr)
    return report


def _validation_report_document(
    raw_report: dict[str, object],
    *,
    run_id: str,
    tolerance: float,
) -> dict[str, object]:
    latency = raw_report.get("latency")
    status = _validation_status(raw_report.get("status"))
    document = {
        "schema_version": 1,
        "status": status,
        "validation_run_id": run_id,
        "model_uri": f"runs:/{run_id}/model",
        "row_count": int(raw_report.get("row_count") or 0),
        "max_abs_diff": raw_report.get("max_abs_diff"),
        "tolerance": raw_report.get("tolerance", tolerance),
        "checks": raw_report.get("checks", {}),
        "failed_checks": raw_report.get("failed_checks", []),
        "latency": latency if isinstance(latency, dict) else {},
        "latency_ms": (
            latency.get("latency_ms", {}) if isinstance(latency, dict) else {}
        ),
        "files": {
            "input_csv": "validation/data/input.csv",
            "input_parquet": "validation/data/input.parquet",
            "expected_parquet": "validation/data/expected.parquet",
            "latency_detail": "validation/latency_detail.json",
        },
    }
    if raw_report.get("error"):
        document["error"] = raw_report["error"]
    for key in ("error_class", "signal", "signal_name", "stderr_tail"):
        if raw_report.get(key) not in (None, ""):
            document[key] = raw_report[key]
    return document


def _validation_status(value: object) -> str:
    normalized = str(value or "unknown").lower()
    if normalized in {"success", "passed", "pass"}:
        return "success"
    if normalized in {"failed", "fail", "failure"}:
        return "failed"
    if normalized in {"warning", "warn"}:
        return "warning"
    return "error"


def _log_validation_tags_and_metrics(run_id: str, report: dict[str, object]) -> None:
    status = str(report.get("status") or "unknown")
    mlflow_trial.set_tags(run_id, {mlflow_tags.VALIDATION_STATUS: status})
    metrics: dict[str, float] = {}
    max_abs_diff = report.get("max_abs_diff")
    if max_abs_diff is not None:
        metrics["validation.max_abs_diff"] = float(max_abs_diff)
    latency_ms = report.get("latency_ms")
    if isinstance(latency_ms, dict):
        if "median" in latency_ms:
            metrics["validation.latency_p50_ms"] = float(latency_ms["median"])
        if "p99" in latency_ms:
            metrics["validation.latency_p99_ms"] = float(latency_ms["p99"])
    if metrics:
        mlflow_trial.log_metrics(run_id, metrics)
