"""Shared safe slug primitives."""

from __future__ import annotations

import re


SLUG_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


__all__ = ["SLUG_RE"]
