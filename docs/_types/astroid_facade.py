"""Typed helpers for loading Astroid manager and builder classes."""

from __future__ import annotations

import inspect
from types import ModuleType
from typing import Protocol, cast, runtime_checkable

__all__ = [
    "AstroidBuilderProtocol",
    "AstroidManagerProtocol",
    "coerce_astroid_builder_class",
    "coerce_astroid_manager_class",
]

_MISSING = object()


@runtime_checkable
class AstroidManagerProtocol(Protocol):
    """Subset of :mod:`astroid.manager` consumed by the docs build."""

    def build_from_file(self, path: str) -> object:
        """Return the AST for ``path``."""


@runtime_checkable
class AstroidBuilderProtocol(Protocol):
    """Subset of :mod:`astroid.builder` consumed by the docs build."""

    def __init__(self, manager: AstroidManagerProtocol | None = None) -> None:
        """Initialise the builder with an optional manager instance."""

    def file_build(self, file_path: str, module_name: str) -> object:
        """Return an AST node representing ``module_name`` at ``file_path``."""


def _coerce_class(module: ModuleType, attribute: str, kind: str) -> type[object]:
    candidate_obj = getattr(module, attribute, _MISSING)
    if candidate_obj is _MISSING or not inspect.isclass(candidate_obj):
        message = f"Module '{module.__name__}' attribute '{attribute}' is not a {kind} class"
        raise TypeError(message)
    return cast(type[object], candidate_obj)


def coerce_astroid_manager_class(module: ModuleType) -> type[AstroidManagerProtocol]:
    """Return the typed Astroid manager class from ``module``."""
    manager_cls = _coerce_class(module, "AstroidManager", "AstroidManager")
    return cast(type[AstroidManagerProtocol], manager_cls)


def coerce_astroid_builder_class(module: ModuleType) -> type[AstroidBuilderProtocol]:
    """Return the typed Astroid builder class from ``module``."""
    builder_cls = _coerce_class(module, "AstroidBuilder", "AstroidBuilder")
    return cast(type[AstroidBuilderProtocol], builder_cls)
