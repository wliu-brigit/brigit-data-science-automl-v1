"""Trial data-contract artifact writer."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
import tempfile
from pathlib import Path
from typing import Any, Mapping

from automl.errors import StorageError
from automl.mlflow import _routing
from automl.mlflow import client
from automl.mlflow import tags
from automl.mlflow.trial.logging import log_json, set_tag
from automl.utils.io import gcs as _gcs


@dataclass(frozen=True)
class TrialDataContractRef:
    run_id: str
    uri: str
    path: str = "data/contract.json"


def write_trial_data_contract(run_id: str, payload: object) -> TrialDataContractRef:
    document = _payload_to_dict(payload)
    path = "data/contract.json"
    uri = _write_payload(run_id, path, document)
    try:
        set_tag(run_id, tags.DATA_CONTRACT_URI, path)
    except Exception as exc:
        raise StorageError("Failed to commit trial data contract artifact") from exc
    return TrialDataContractRef(run_id=run_id, uri=uri)


def load_trial_data_contract(run_id: str):
    from automl.data.contract import TrialDataContract

    run_tags = client.raw().get_run(run_id).data.tags
    uri = run_tags.get(tags.DATA_CONTRACT_URI)
    if not uri:
        raise StorageError(f"run {run_id!r} is missing {tags.DATA_CONTRACT_URI!r}")
    uri = _normalize_run_artifact_uri(run_id, uri)
    if uri.startswith("gs://"):
        try:
            return TrialDataContract.from_dict(_gcs.read_json(uri))
        except Exception as exc:
            raise StorageError(f"Failed to read trial data contract {uri!r}") from exc
    if uri.startswith(f"runs:/{run_id}/"):
        path = uri.removeprefix(f"runs:/{run_id}/")
        try:
            local_path = client.download_artifact(run_id, path, required=True)
            with open(local_path, encoding="utf-8") as handle:
                return TrialDataContract.from_dict(json.load(handle))
        except Exception as exc:
            raise StorageError(f"Failed to read trial data contract {uri!r}") from exc
    raise StorageError(f"unsupported trial data contract URI {uri!r}")


def _payload_to_dict(payload: object) -> dict[str, Any]:
    if hasattr(payload, "to_dict") and callable(payload.to_dict):
        document = dict(payload.to_dict())
    elif is_dataclass(payload) and not isinstance(payload, type):
        document = asdict(payload)
    elif isinstance(payload, Mapping):
        document = dict(payload)
    else:
        raise TypeError("artifact payload must be a mapping, dataclass, or expose to_dict()")
    document.setdefault("schema_version", 1)
    return document


def _write_payload(
    run_id: str,
    path: str,
    payload: dict[str, Any],
    *,
    overwrite: bool = False,
) -> str:
    del overwrite
    log_json(run_id, path, payload)
    return f"runs:/{run_id}/{path}"


def _write_bytes_payload(
    run_id: str,
    path: str,
    payload: bytes,
    *,
    content_type: str = "application/octet-stream",
    overwrite: bool = False,
) -> str:
    del overwrite

    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_path = Path(path)
        local_path = Path(tmp_dir) / artifact_path.name
        local_path.write_bytes(payload)
        artifact_parent = artifact_path.parent.as_posix()
        try:
            client.raw().log_artifact(
                run_id,
                str(local_path),
                artifact_path=artifact_parent if artifact_parent != "." else None,
            )
        except Exception as exc:
            raise StorageError(f"Failed to log binary artifact {path!r}") from exc
    return f"runs:/{run_id}/{path}"


def _write_large_bytes_payload(
    run_id: str,
    path: str,
    payload: bytes,
    *,
    content_type: str = "application/octet-stream",
    overwrite: bool = False,
) -> str:
    if client.bound().bucket:
        uri = _routing.bucket_uri_for(kind="run_bulk", run_id=run_id) + path
        try:
            _gcs.write_bytes(uri, payload, content_type=content_type, overwrite=overwrite)
        except Exception as exc:
            raise StorageError(f"Failed to write GCS artifact {uri!r}") from exc
        return uri
    return _write_bytes_payload(
        run_id,
        path,
        payload,
        content_type=content_type,
        overwrite=overwrite,
    )


def _normalize_run_artifact_uri(run_id: str, value: str) -> str:
    if value.startswith(("gs://", f"runs:/{run_id}/")):
        return value
    return f"runs:/{run_id}/{value.strip('/')}"


__all__ = ["TrialDataContractRef", "load_trial_data_contract", "write_trial_data_contract"]
