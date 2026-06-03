"""Per-label prediction artifact writer and loader."""

from __future__ import annotations

import io
from dataclasses import dataclass

import pandas as pd

from automl.errors import StorageError
from automl.eval.results import Predictions
from automl.mlflow import client, tags
from automl.mlflow.trial.artifacts.data import _write_large_bytes_payload, _write_payload
from automl.mlflow.trial.artifacts.eval import validate_eval_label
from automl.mlflow.trial.logging import set_tags
from automl.utils.io import gcs as _gcs


@dataclass(frozen=True)
class PredictionsRef:
    run_id: str
    label: str
    uri: str
    manifest_uri: str
    path: str
    manifest_path: str


def write_predictions(
    run_id: str,
    label: str,
    payload: Predictions,
    *,
    overwrite: bool = False,
) -> PredictionsRef:
    safe_label = validate_eval_label(label)
    path = f"eval/{safe_label}/predictions.parquet"
    manifest_path = f"eval/{safe_label}/predictions.json"
    parquet_uri = _write_large_bytes_payload(
        run_id,
        path,
        _frame_to_parquet(payload.frame),
        content_type="application/octet-stream",
        overwrite=overwrite,
    )
    manifest_uri = _write_payload(
        run_id,
        manifest_path,
        payload.manifest_dict(),
        overwrite=overwrite,
    )
    try:
        set_tags(
            run_id,
            {
                tags.eval_predictions_uri(safe_label): parquet_uri,
                tags.eval_predictions_manifest_uri(safe_label): manifest_path,
            },
        )
    except Exception as exc:
        raise StorageError("Failed to commit predictions artifact") from exc
    return PredictionsRef(
        run_id=run_id,
        label=safe_label,
        uri=parquet_uri,
        manifest_uri=manifest_uri,
        path=path,
        manifest_path=manifest_path,
    )


def load_predictions(run_id: str, label: str) -> Predictions:
    safe_label = validate_eval_label(label)
    run_tags = client.raw().get_run(run_id).data.tags
    parquet_uri = run_tags.get(tags.eval_predictions_uri(safe_label))
    manifest_uri = run_tags.get(tags.eval_predictions_manifest_uri(safe_label))
    if not parquet_uri or not manifest_uri:
        raise StorageError(
            f"run {run_id!r} is missing predictions artifact for label {safe_label!r}"
        )
    manifest = _read_json(run_id, manifest_uri)
    frame = pd.read_parquet(io.BytesIO(_read_bytes(run_id, parquet_uri)))
    return Predictions.from_parts(manifest, frame)


def list_predictions(run_id: str) -> list[str]:
    run_tags = client.raw().get_run(run_id).data.tags
    prefix = "eval."
    suffix = ".predictions_uri"
    labels = []
    for key in run_tags:
        if key.startswith(prefix) and key.endswith(suffix):
            labels.append(key.removeprefix(prefix).removesuffix(suffix))
    return sorted(labels)


def _frame_to_parquet(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False, compression="snappy")
    return buffer.getvalue()


def _read_json(run_id: str, uri: str) -> dict:
    from automl.mlflow.trial.artifacts.eval import _read_json_uri

    return _read_json_uri(run_id, uri)


def _read_bytes(run_id: str, uri: str) -> bytes:
    if uri.startswith("gs://"):
        try:
            return _gcs.read_bytes(uri)
        except Exception as exc:
            raise StorageError(f"Failed to read binary artifact {uri!r}") from exc
    if uri.startswith(f"runs:/{run_id}/"):
        path = uri.removeprefix(f"runs:/{run_id}/")
        try:
            local_path = client.download_artifact(run_id, path, required=True)
            with open(local_path, "rb") as handle:
                return handle.read()
        except Exception as exc:
            raise StorageError(f"Failed to read binary artifact {uri!r}") from exc
    raise StorageError(f"unsupported artifact URI {uri!r}")


__all__ = [
    "PredictionsRef",
    "list_predictions",
    "load_predictions",
    "write_predictions",
]
