"""Utilities for documentation scripts with lazy accessors."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, cast

__all__ = ["validate_against_schema"]


def __getattr__(name: str) -> object:
    if name == "validate_against_schema":
        module = import_module("docs.scripts.validation")
        return cast(object, getattr(module, name))
    message = f"module '{__name__}' has no attribute '{name}'"
    raise AttributeError(message)


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # pragma: no cover - typing assistance only
    from docs.scripts.validation import validate_against_schema
