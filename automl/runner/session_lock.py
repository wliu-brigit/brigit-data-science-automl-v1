"""Runner-owned filesystem lock for one project route."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from automl.mlflow import routing as mlflow_routing


DEFAULT_STALE_AFTER_SECONDS = 21_600
LOCK_ERROR = 4


def lock_dir(project_root: Path) -> Path:
    return Path(project_root) / ".cache" / "automl" / "tmp" / "session_locks"


def lock_path(project_root: Path, route: str) -> Path:
    return lock_dir(Path(project_root)) / f"{_route_key(route)}.lock"


def is_locked(*, project_root: Path, route: str) -> bool:
    return lock_path(Path(project_root), route).exists()


def route_for_session(active: Any) -> str:
    return mlflow_routing.experiment_route_for(
        project_name=str(active.project_name),
        experiment_id=str(active.active_experiment_id),
        namespace=str(getattr(active, "namespace", "") or ""),
        dry_run=bool(getattr(active, "dry_run", False)),
    )


def acquire_for_session(active: Any, session_id: str) -> dict[str, str]:
    route = route_for_session(active)
    lock_id = acquire(
        project_root=active.config.repo_root,
        route=route,
        session_id=session_id,
    )
    return {
        "status": "acquired",
        "session_id": session_id,
        "route": route,
        "lock_id": lock_id,
    }


def release_for_session(active: Any, session_id: str, lock_id: str) -> dict[str, str]:
    release(
        project_root=active.config.repo_root,
        session_id=session_id,
        lock_id=lock_id,
    )
    return {"status": "released", "session_id": session_id, "lock_id": lock_id}


def acquire(
    *,
    project_root: Path,
    route: str,
    session_id: str,
    stale_after_seconds: int = DEFAULT_STALE_AFTER_SECONDS,
) -> str:
    """Acquire the session lock and return its lock id."""

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        code = _acquire_exit_code(
            Path(project_root),
            route=route,
            session_id=session_id,
            stale_after_seconds=stale_after_seconds,
        )
    if code != 0:
        detail = output.getvalue().strip()
        raise RuntimeError(detail or f"session lock acquire failed with code {code}")
    payload_path = _metadata_path(lock_path(Path(project_root), route))
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    lock_id = str(payload.get("lock_id") or "")
    if not lock_id:
        raise RuntimeError("session lock acquired without a lock_id")
    return lock_id


def release(*, project_root: Path, session_id: str, lock_id: str) -> None:
    """Release a previously acquired lock."""

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        code = _release_exit_code(Path(project_root), session_id=session_id, lock_id=lock_id)
    if code != 0:
        detail = output.getvalue().strip()
        raise RuntimeError(detail or f"session lock release failed with code {code}")


@contextlib.contextmanager
def session_lock(
    *,
    project_root: Path,
    route: str,
    session_id: str,
    stale_after_seconds: int = DEFAULT_STALE_AFTER_SECONDS,
) -> Iterator[str]:
    lock_id = acquire(
        project_root=Path(project_root),
        route=route,
        session_id=session_id,
        stale_after_seconds=stale_after_seconds,
    )
    try:
        yield lock_id
    finally:
        release(project_root=Path(project_root), session_id=session_id, lock_id=lock_id)


def _route_key(route: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", route.strip()).strip("_")
    digest = hashlib.sha256(route.encode("utf-8")).hexdigest()[:8]
    return f"{normalized or 'route'}_{digest}"


def _metadata_path(path: Path) -> Path:
    return path / "metadata.json"


def _read_lock(path: Path) -> dict[str, Any]:
    payload = json.loads(_metadata_path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"lock payload is not a JSON object: {path}")
    float(payload.get("created_at", time.time()))
    float(payload.get("stale_after_seconds", 0))
    return payload


def _write_lock_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path / f"metadata.{os.getpid()}.tmp"
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, _metadata_path(path))


def _remove_lock(path: Path) -> None:
    shutil.rmtree(path)


def _recovery_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.recovery")


def _try_enter_recovery(path: Path) -> bool:
    try:
        _recovery_path(path).mkdir(mode=0o700)
    except FileExistsError:
        return False
    return True


def _exit_recovery(path: Path) -> None:
    shutil.rmtree(_recovery_path(path), ignore_errors=True)


def _lock_age_seconds(path: Path, now: float) -> float | None:
    try:
        return now - path.stat().st_mtime
    except FileNotFoundError:
        return None


def _try_recover_lock(
    path: Path,
    *,
    route: str,
    stale_after_seconds: int,
    reason: str,
) -> str:
    if not _try_enter_recovery(path):
        print(f"LOCKED: route={route} session=<recovering>")
        return "locked"
    try:
        now = time.time()
        try:
            existing = _read_lock(path)
        except FileNotFoundError:
            age = _lock_age_seconds(path, now)
            if age is None:
                return "retry"
            if age <= stale_after_seconds:
                print(f"LOCKED: route={route} session=<initializing>")
                return "locked"
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        else:
            created_at = float(existing.get("created_at", now))
            stale_after = float(existing.get("stale_after_seconds", stale_after_seconds))
            if now - created_at <= stale_after:
                print(
                    "LOCKED: "
                    f"route={existing.get('route')} "
                    f"session={existing.get('session_id')}"
                )
                return "locked"
        try:
            _remove_lock(path)
        except FileNotFoundError:
            return "retry"
        except OSError as exc:
            print(f"LOCK_ERROR: {reason} lock removal failed: {exc}")
            return "error"
        if reason == "corrupt":
            print("STALE_CORRUPT_LOCK_REPLACED")
        return "retry"
    finally:
        _exit_recovery(path)


def _acquire_exit_code(
    project_root: Path,
    *,
    route: str,
    session_id: str,
    stale_after_seconds: int,
) -> int:
    path = lock_path(project_root, route)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_id = uuid.uuid4().hex
    payload = {
        "lock_id": lock_id,
        "session_id": session_id,
        "route": route,
        "created_at": time.time(),
        "stale_after_seconds": stale_after_seconds,
    }
    while True:
        try:
            path.mkdir(mode=0o700)
            _write_lock_atomic(path, payload)
            print(f"ACQUIRED: route={route} session={session_id} lock_id={lock_id}")
            return 0
        except FileExistsError:
            now = time.time()
            try:
                existing = _read_lock(path)
            except FileNotFoundError:
                age = _lock_age_seconds(path, now)
                if age is None:
                    continue
                if age <= stale_after_seconds:
                    print(f"LOCKED: route={route} session=<initializing>")
                    return 2
                recovery_status = _try_recover_lock(
                    path,
                    route=route,
                    stale_after_seconds=stale_after_seconds,
                    reason="initializing",
                )
                if recovery_status == "retry":
                    continue
                return LOCK_ERROR if recovery_status == "error" else 2
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                recovery_status = _try_recover_lock(
                    path,
                    route=route,
                    stale_after_seconds=stale_after_seconds,
                    reason="corrupt",
                )
                if recovery_status == "retry":
                    continue
                if recovery_status == "error":
                    return LOCK_ERROR
                print(f"LOCKED: route={route} session=<recovering> error={exc}")
                return 2
            created_at = float(existing.get("created_at", now))
            stale_after = float(existing.get("stale_after_seconds", stale_after_seconds))
            if now - created_at <= stale_after:
                print(
                    "LOCKED: "
                    f"route={existing.get('route')} "
                    f"session={existing.get('session_id')}"
                )
                return 2
            recovery_status = _try_recover_lock(
                path,
                route=route,
                stale_after_seconds=stale_after_seconds,
                reason="stale",
            )
            if recovery_status == "retry":
                continue
            return LOCK_ERROR if recovery_status == "error" else 2


def _release_exit_code(project_root: Path, *, session_id: str, lock_id: str) -> int:
    root = lock_dir(project_root)
    if not root.exists():
        print("NO_LOCK")
        return 0
    owned_by_other: list[str] = []
    owned_by_session = False
    busy = False
    for path in sorted(root.glob("*.lock")):
        if not _try_enter_recovery(path):
            busy = True
            continue
        try:
            try:
                payload = _read_lock(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if payload.get("session_id") != session_id:
                owner = payload.get("session_id")
                route = payload.get("route")
                owned_by_other.append(f"{route}:{owner}")
                continue
            owned_by_session = True
            if payload.get("lock_id") != lock_id:
                continue
            _remove_lock(path)
            print(f"RELEASED: route={payload.get('route')} session={session_id}")
            return 0
        finally:
            _exit_recovery(path)
    if busy:
        print("LOCK_BUSY")
        return 2
    if owned_by_session:
        print("LOCK_ID_MISMATCH")
        return 3
    if owned_by_other:
        print(f"LOCK_OWNED_BY_OTHER: {', '.join(owned_by_other)}")
        return 3
    print("NO_LOCK")
    return 0


__all__ = [
    "DEFAULT_STALE_AFTER_SECONDS",
    "LOCK_ERROR",
    "acquire",
    "acquire_for_session",
    "is_locked",
    "lock_dir",
    "lock_path",
    "release",
    "release_for_session",
    "route_for_session",
    "session_lock",
]
