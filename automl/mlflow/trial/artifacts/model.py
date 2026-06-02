"""Model artifact writers for MLflow-visible trial models."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
import sys
from typing import Any

import cloudpickle
import mlflow.pyfunc

from automl.errors import StorageError
from automl.mlflow import client
from automl.mlflow import tags
from automl.mlflow.trial.artifacts.data import _normalize_run_artifact_uri, _write_bytes_payload
from automl.mlflow.trial.logging import set_tag
from automl.utils.io import gcs as _gcs


@dataclass(frozen=True)
class ModelRef:
    run_id: str
    uri: str
    path: str = "model"
    # MLflow 3 logged-model URI (``models:/<model_id>``), when the backend
    # records one. Prefer this for loads: it resolves straight to the model's
    # artifact location instead of probing the (empty) run artifact path.
    logged_uri: str | None = None


class _AutoMLPyfuncWrapper(mlflow.pyfunc.PythonModel):
    # MLflow inspects PythonModel.predict type hints at subclass definition time.
    # This adapter is deliberately generic; signatures are supplied at log time.
    _skip_type_hint_validation = True

    def __init__(self, model: object) -> None:
        self.model = model

    def predict(self, context, model_input, params: dict[str, Any] | None = None):
        model_input = _cast_model_input(self.model, model_input)
        predict = getattr(self.model, "predict")
        try:
            return predict(context=None, model_input=model_input, params=params)
        except TypeError:
            try:
                return predict(context=None, model_input=model_input)
            except TypeError:
                return predict(model_input)


def _cast_model_input(model: object, model_input):
    registry = getattr(model, "feature_registry", None)
    if registry is None or not hasattr(registry, "cast"):
        return model_input
    if not hasattr(model_input, "copy"):
        return model_input
    try:
        return registry.cast(model_input.copy(), inplace=True)
    except Exception:
        return model_input


def write_model(
    run_id: str,
    payload: object,
    *,
    input_example=None,
    signature=None,
    code_paths: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ModelRef:
    path = "model"
    uri = f"runs:/{run_id}/{path}"
    signature = signature or _permissive_signature(input_example)
    try:
        import mlflow

        mlflow.set_tracking_uri(client.bound().tracking_uri)
        active = mlflow.active_run()

        def _log() -> Any:
            return mlflow.pyfunc.log_model(
                artifact_path=path,
                python_model=_AutoMLPyfuncWrapper(payload),
                input_example=input_example,
                signature=signature,
                code_paths=code_paths,
                metadata=metadata,
            )

        with _suppress_bytecode_writes():
            if active is not None and active.info.run_id == run_id:
                model_info = _log()
            else:
                with mlflow.start_run(run_id=run_id):
                    model_info = _log()
    except Exception as exc:
        raise StorageError("Failed to log MLflow pyfunc model") from exc
    logged_uri = _logged_model_uri(model_info)
    try:
        set_tag(run_id, tags.MODEL_URI, uri)
        if logged_uri is not None:
            set_tag(run_id, tags.MODEL_LOGGED_ID, _model_id(model_info))
    except Exception as exc:
        raise StorageError("Failed to commit model artifact") from exc
    return ModelRef(run_id=run_id, uri=uri, path=path, logged_uri=logged_uri)


def _model_id(model_info: Any) -> str | None:
    """Best-effort extraction of the MLflow 3 logged-model id from a ModelInfo."""
    model_id = getattr(model_info, "model_id", None)
    return str(model_id) if model_id else None


def _logged_model_uri(model_info: Any) -> str | None:
    """Return ``models:/<model_id>`` when the backend records a logged model."""
    model_id = _model_id(model_info)
    return f"models:/{model_id}" if model_id else None


def logged_model_uri_for(run_id: str) -> str | None:
    """Resolve a run's logged-model URI from its persisted tag, if any.

    Deterministic from the run alone — no dependence on the ``log_model``
    return value at the call site.
    """
    run_tags = client.raw().get_run(run_id).data.tags
    model_id = run_tags.get(tags.MODEL_LOGGED_ID)
    return f"models:/{model_id}" if model_id else None


def write_pickle_model(run_id: str, payload: object) -> ModelRef:
    path = "model.pkl"
    uri = _write_bytes_payload(run_id, path, _model_payload_bytes(payload))
    try:
        set_tag(run_id, tags.MODEL_URI, uri)
    except Exception as exc:
        raise StorageError("Failed to commit model artifact") from exc
    return ModelRef(run_id=run_id, uri=uri, path=path)


def load_model(run_id: str):
    run_tags = client.raw().get_run(run_id).data.tags
    # Prefer the MLflow 3 logged-model URI when present: ``models:/<id>``
    # resolves directly to the model's artifact location and skips the legacy
    # ``runs:/<run>/model`` probe (which 500s + retries when absent there).
    logged_id = run_tags.get(tags.MODEL_LOGGED_ID)
    if logged_id:
        logged_uri = f"models:/{logged_id}"
        try:
            import mlflow

            mlflow.set_tracking_uri(client.bound().tracking_uri)
            return mlflow.pyfunc.load_model(logged_uri)
        except Exception as exc:
            raise StorageError(f"Failed to read model artifact {logged_uri!r}") from exc
    uri = run_tags.get(tags.MODEL_URI)
    if not uri:
        raise StorageError(f"run {run_id!r} is missing {tags.MODEL_URI!r}")
    uri = _normalize_run_artifact_uri(run_id, uri)
    if uri.endswith("/model"):
        try:
            import mlflow

            mlflow.set_tracking_uri(client.bound().tracking_uri)
            return mlflow.pyfunc.load_model(uri)
        except Exception as exc:
            raise StorageError(f"Failed to read model artifact {uri!r}") from exc
    if uri.startswith("gs://"):
        try:
            return cloudpickle.loads(_gcs.read_bytes(uri))
        except Exception as exc:
            raise StorageError(f"Failed to read model artifact {uri!r}") from exc
    if uri.startswith(f"runs:/{run_id}/"):
        path = uri.removeprefix(f"runs:/{run_id}/")
        try:
            local_path = client.raw().download_artifacts(run_id, path)
            with open(local_path, "rb") as handle:
                return cloudpickle.loads(handle.read())
        except Exception as exc:
            raise StorageError(f"Failed to read model artifact {uri!r}") from exc
    raise StorageError(f"unsupported model artifact URI {uri!r}")


def load_model_source(run_id: str) -> str:
    run_tags = client.raw().get_run(run_id).data.tags
    uri = run_tags.get(tags.MODEL_SOURCE_URI)
    if not uri:
        return _load_model_source_from_pyfunc_code(run_id)
    uri = _normalize_run_artifact_uri(run_id, uri)
    if uri.startswith("gs://"):
        try:
            return _gcs.read_bytes(uri).decode("utf-8")
        except Exception as exc:
            raise StorageError(f"Failed to read model source artifact {uri!r}") from exc
    if uri.startswith(f"runs:/{run_id}/"):
        path = uri.removeprefix(f"runs:/{run_id}/")
        try:
            local_path = client.raw().download_artifacts(run_id, path)
            with open(local_path, encoding="utf-8") as handle:
                return handle.read()
        except Exception as exc:
            raise StorageError(f"Failed to read model source artifact {uri!r}") from exc
    raise StorageError(f"unsupported model source artifact URI {uri!r}")


def _permissive_signature(input_example):
    if input_example is None or not hasattr(input_example, "columns"):
        return None
    try:
        from mlflow.models.signature import ModelSignature
        from mlflow.types import ColSpec, Schema
        from mlflow.types.schema import AnyType
    except Exception:
        return None
    return ModelSignature(
        inputs=Schema(
            [
                ColSpec(AnyType(), name=str(column), required=False)
                for column in input_example.columns
            ]
        )
    )


@contextmanager
def _suppress_bytecode_writes():
    old_value = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.dont_write_bytecode = old_value


def _load_model_source_from_pyfunc_code(run_id: str) -> str:
    candidates = _walk_model_code_artifacts(run_id, "model/code")
    preferred = _preferred_source_path(run_id, candidates)
    if preferred is None:
        raise StorageError(
            f"run {run_id!r} is missing {tags.MODEL_SOURCE_URI!r} and pyfunc code source"
        )
    try:
        local_path = client.raw().download_artifacts(run_id, preferred)
        with open(local_path, encoding="utf-8") as handle:
            return handle.read()
    except Exception as exc:
        raise StorageError(f"Failed to read model source artifact {preferred!r}") from exc


def _walk_model_code_artifacts(run_id: str, path: str) -> list[str]:
    paths: list[str] = []
    try:
        items = client.raw().list_artifacts(run_id, path)
    except Exception:
        return paths
    for item in items:
        item_path = str(item.path)
        if getattr(item, "is_dir", False):
            paths.extend(_walk_model_code_artifacts(run_id, item_path))
        else:
            paths.append(item_path)
    return paths


def _preferred_source_path(run_id: str, paths: list[str]) -> str | None:
    run_tags = client.raw().get_run(run_id).data.tags
    project_name = str(run_tags.get(tags.PROJECT_NAME, ""))
    project_model_suffix = (
        f"/projects/{project_name}/model/__init__.py" if project_name else ""
    )
    for path in paths:
        if Path(path).name.startswith("trial_model_") and path.endswith(".py"):
            return path
    for path in paths:
        if project_model_suffix and path.endswith(project_model_suffix):
            return path
    for path in paths:
        if path.endswith("/model.py"):
            return path
    for path in paths:
        if path.endswith("/model/__init__.py"):
            return path
    return None


def _model_payload_bytes(payload: object) -> bytes:
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, bytearray):
        return bytes(payload)
    if isinstance(payload, Path):
        return payload.read_bytes()
    if isinstance(payload, str):
        path = Path(payload)
        if path.exists():
            return path.read_bytes()
        if _looks_path_like(payload):
            raise FileNotFoundError(path)
    return cloudpickle.dumps(payload)


def _looks_path_like(value: str) -> bool:
    return (
        "/" in value
        or "\\" in value
        or value.startswith(".")
        or Path(value).suffix == ".pkl"
    )


__all__ = [
    "ModelRef",
    "load_model",
    "load_model_source",
    "logged_model_uri_for",
    "write_model",
    "write_pickle_model",
]
