"""Dataset recipe: the config-derived identity of a materialization.

The recipe answers "SHOULD the dataset be different?" from config alone,
without touching any source. Its field list is a mechanical rule — the
transitive set of inputs materialize() reads — not a curated list.
Content identity (identity_hash) remains the only dedup key; the recipe is
recorded on the dataset record so drift reports can name fields.
"""

from __future__ import annotations

from typing import Any, Mapping


def compute_recipe(spec: Any, session: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "source": spec.source.recipe_identity(project_dir=session.config.project_dir),
        "exclude_cols": sorted(spec.exclude_cols),
        "metadata_cols": sorted(spec.metadata_cols),
        "null_drop_threshold": float(spec.null_drop_threshold),
        "constant_drop_threshold": float(spec.constant_drop_threshold),
        "pipeline_cls": f"{spec.pipeline_cls.__module__}.{spec.pipeline_cls.__qualname__}",
        "target": session.config.raw_target_column,
    }
    if session.dry_run:
        payload["dry_run_rows"] = int(spec.dry_run_rows)
    return payload


def recipe_diff(
    recorded: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    prefix: str = "",
) -> list[str]:
    """Dotted paths of fields that differ — the payload of a drift warning."""
    fields: list[str] = []
    for key in sorted(set(recorded) | set(current)):
        left, right = recorded.get(key), current.get(key)
        path = f"{prefix}{key}"
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            fields.extend(recipe_diff(left, right, prefix=f"{path}."))
        elif left != right:
            fields.append(path)
    return fields


__all__ = ["compute_recipe", "recipe_diff"]
