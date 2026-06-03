"""Per-label eval artifact writer."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from automl.errors import StorageError
from automl.eval.results import EvalIndex, EvalResult
from automl.mlflow import client
from automl.mlflow import tags
from automl.mlflow.trial.artifacts.data import (
    _normalize_run_artifact_uri,
    _payload_to_dict,
    _write_payload,
)
from automl.mlflow.trial.logging import set_tag
from automl.utils.io import gcs as _gcs


_LABEL_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class EvalRef:
    run_id: str
    label: str
    uri: str
    path: str


@dataclass(frozen=True)
class EvalIndexRef:
    run_id: str
    uri: str
    path: str = "eval/manifest.json"


def write_eval(
    run_id: str,
    label: str,
    payload: object,
    *,
    overwrite: bool = False,
) -> EvalRef:
    safe_label = validate_eval_label(label)
    document = _payload_to_dict(payload)
    path = f"eval/{safe_label}/report.json"
    uri = _write_payload(run_id, path, document, overwrite=overwrite)
    try:
        set_tag(run_id, tags.eval_uri(safe_label), path)
        eval_dataset_id = document.get("eval_dataset_id")
        if eval_dataset_id is not None:
            set_tag(run_id, tags.eval_dataset_id(safe_label), eval_dataset_id)
    except Exception as exc:
        raise StorageError("Failed to commit eval artifact") from exc
    return EvalRef(run_id=run_id, label=safe_label, uri=uri, path=path)


def load_eval(run_id: str, label: str) -> EvalResult:
    safe_label = validate_eval_label(label)
    run_tags = client.raw().get_run(run_id).data.tags
    uri = run_tags.get(tags.eval_uri(safe_label))
    if not uri:
        raise StorageError(f"run {run_id!r} is missing eval artifact for label {safe_label!r}")
    return EvalResult.from_dict(_read_json_uri(run_id, uri))


def list_eval(run_id: str) -> list[tuple[str, str]]:
    run_tags = client.raw().get_run(run_id).data.tags
    prefix = "eval."
    suffix = ".dataset_id"
    labels = []
    for key, eval_dataset_id in run_tags.items():
        if key.startswith(prefix) and key.endswith(suffix):
            label = key.removeprefix(prefix).removesuffix(suffix)
            labels.append((label, eval_dataset_id))
    return sorted(labels)


def write_eval_index(run_id: str, payload: object) -> EvalIndexRef:
    document = _payload_to_dict(payload)
    path = "eval/manifest.json"
    uri = _write_payload(run_id, path, document, overwrite=True)
    try:
        set_tag(run_id, tags.EVAL_INDEX_URI, path)
    except Exception as exc:
        raise StorageError("Failed to commit eval index artifact") from exc
    return EvalIndexRef(run_id=run_id, uri=uri)


def load_eval_index(run_id: str) -> EvalIndex:
    run_tags = client.raw().get_run(run_id).data.tags
    uri = run_tags.get(tags.EVAL_INDEX_URI)
    if not uri:
        return EvalIndex(primary_label=None, evaluations=())
    return EvalIndex.from_dict(_read_json_uri(run_id, uri))


def validate_eval_label(label: str) -> str:
    if not isinstance(label, str) or not label or label in {".", ".."}:
        raise ValueError("eval label required")
    if not _LABEL_RE.fullmatch(label):
        raise ValueError("eval label must contain only letters, numbers, '_', '-', or '.'")
    return label


def _read_json_uri(run_id: str, uri: str) -> dict:
    uri = _normalize_run_artifact_uri(run_id, uri)
    if uri.startswith("gs://"):
        try:
            return _gcs.read_json(uri)
        except Exception as exc:
            raise StorageError(f"Failed to read JSON artifact {uri!r}") from exc
    if uri.startswith(f"runs:/{run_id}/"):
        path = uri.removeprefix(f"runs:/{run_id}/")
        try:
            local_path = client.download_artifact(run_id, path, required=True)
            with open(local_path, encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {uri}")
            return payload
        except Exception as exc:
            raise StorageError(f"Failed to read JSON artifact {uri!r}") from exc
    raise StorageError(f"unsupported artifact URI {uri!r}")


__all__ = [
    "EvalIndexRef",
    "EvalRef",
    "list_eval",
    "load_eval",
    "load_eval_index",
    "validate_eval_label",
    "write_eval",
    "write_eval_index",
]
