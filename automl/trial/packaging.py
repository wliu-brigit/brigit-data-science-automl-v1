"""Helpers for turning authored model classes into trial source files."""

from __future__ import annotations

import inspect as pyinspect
import linecache
import re
import textwrap
from pathlib import Path
from typing import Any


def package_model(
    model_class: type[Any],
    *,
    imports: list[str],
    output_path: Path,
) -> Path:
    """Write a class source file exposing ``Model`` for trial execution."""

    if not isinstance(model_class, type):
        raise TypeError("model_class must be a class")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import_block = "\n".join(line.strip() for line in imports if line.strip())
    source = _model_class_source(model_class)
    text = "\n\n".join(part for part in (import_block, source) if part).rstrip() + "\n"
    if model_class.__name__ != "Model":
        text += f"\nModel = {model_class.__name__}\n"
    output_path.write_text(text, encoding="utf-8")
    return output_path


def _model_class_source(model_class: type[Any]) -> str:
    try:
        source = textwrap.dedent(pyinspect.getsource(model_class)).strip()
    except (OSError, TypeError):
        source = ""
    if _source_defines_class(source, model_class.__name__):
        return source

    fallback = _class_source_from_method_code(model_class)
    if fallback:
        return fallback
    raise OSError(f"could not find source for class {model_class.__name__}")


def _source_defines_class(source: str, class_name: str) -> bool:
    pattern = rf"(^|\n)\s*class\s+{re.escape(class_name)}\b"
    return bool(source and re.search(pattern, source))


def _class_source_from_method_code(model_class: type[Any]) -> str:
    for value in model_class.__dict__.values():
        func = _unwrap_function(value)
        if func is None:
            continue
        lines = linecache.getlines(func.__code__.co_filename)
        if not lines:
            continue
        source = _extract_class_block(
            lines,
            class_name=model_class.__name__,
            before_lineno=func.__code__.co_firstlineno,
        )
        if source:
            return source
    return ""


def _unwrap_function(value: Any) -> Any:
    if isinstance(value, (classmethod, staticmethod)):
        value = value.__func__
    return value if pyinspect.isfunction(value) else None


def _extract_class_block(
    lines: list[str],
    *,
    class_name: str,
    before_lineno: int,
) -> str:
    class_pattern = re.compile(rf"^(\s*)class\s+{re.escape(class_name)}\b")
    start = -1
    class_indent = 0
    for index in range(min(before_lineno - 1, len(lines) - 1), -1, -1):
        match = class_pattern.match(lines[index])
        if match:
            start = index
            class_indent = len(match.group(1))
            break
    if start < 0:
        return ""

    while start > 0 and lines[start - 1].lstrip().startswith("@"):
        start -= 1

    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if indent <= class_indent and not line.lstrip().startswith("#"):
            end = index
            break
    return textwrap.dedent("".join(lines[start:end])).strip()


__all__ = ["package_model"]
