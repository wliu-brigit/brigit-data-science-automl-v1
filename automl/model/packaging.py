"""Model serialization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cloudpickle


def save_model(model: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as file_obj:
        cloudpickle.dump(model, file_obj)


__all__ = ["save_model"]
