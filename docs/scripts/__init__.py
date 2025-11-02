"""Public facade for documentation build scripts.

This package provides stable import paths for the internal ``docs._scripts``
modules used by Sphinx tooling and automated tests. Callers should depend on
this package instead of importing the private ``docs._scripts`` namespace
directly.
"""

from __future__ import annotations

import sys
from importlib import import_module
from types import ModuleType
from typing import Final

_SUBMODULES: Final[dict[str, str]] = {
    "shared": "docs._scripts.shared",
    "validation": "docs._scripts.validation",
    "build_symbol_index": "docs._scripts.build_symbol_index",
    "symbol_delta": "docs._scripts.symbol_delta",
    "validate_artifacts": "docs._scripts.validate_artifacts",
    "mkdocs_gen_api": "docs._scripts.mkdocs_gen_api",
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
