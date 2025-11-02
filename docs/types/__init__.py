"""Public facade for documentation type definitions.

This package exposes stable import paths for the internal ``docs._types``
modules that house the authoritative artifact models and optional dependency
facades. Downstream code should depend on these re-exports instead of the
private ``docs._types`` implementations.
"""

from __future__ import annotations

import sys
from importlib import import_module
from types import ModuleType
from typing import Final

_SUBMODULES: Final[dict[str, str]] = {
    "artifacts": "docs._types.artifacts",
    "griffe": "docs._types.griffe",
    "sphinx_optional": "docs._types.sphinx_optional",
}

__all__: tuple[str, ...] = tuple(sorted(_SUBMODULES))


def _load_submodule(name: str) -> ModuleType:
    module = import_module(_SUBMODULES[name])
    sys.modules.setdefault(f"{__name__}.{name}", module)
    return module


def __getattr__(name: str) -> ModuleType:
    if name in _SUBMODULES:
        return _load_submodule(name)
    message = f"module '{__name__}' has no attribute '{name}'"
    raise AttributeError(message)


def __dir__() -> list[str]:
    namespace: dict[str, object] = globals()
    namespace_keys: set[str] = set(namespace.keys())
    return sorted({*namespace_keys, *__all__})
