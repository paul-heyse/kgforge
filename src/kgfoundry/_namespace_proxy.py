"""Internal helpers for namespace bridge modules.

This module is an implementation detail. Downstream code should import
``kgfoundry.tooling_bridge`` instead of relying on these helpers directly.
"""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from types import ModuleType
from typing import Any, cast


def namespace_getattr(module: ModuleType, name: str) -> object:
    """Return ``name`` from ``module`` while preserving the original attribute."""
    return getattr(module, name)


def namespace_exports(module: ModuleType) -> list[str]:
    """Return the public export list for ``module`` respecting ``__all__``."""
    exports: Any = getattr(module, "__all__", None)
    if isinstance(exports, (list, tuple, set)):
        return [str(item) for item in exports]
    return [attr for attr in dir(module) if not attr.startswith("_")]


def namespace_attach(
    module: ModuleType,
    target: MutableMapping[str, object],
    names: Iterable[str],
) -> None:
    """Populate ``target`` with attributes sourced from ``module``."""
    for name in names:
        target[name] = cast(object, getattr(module, name))


def namespace_dir(module: ModuleType, exports: Iterable[str]) -> list[str]:
    """Return the combined attribute listing exposed by a namespace bridge."""
    exported = set(exports)
    exported.update(attr for attr in dir(module) if not attr.startswith("__"))
    return sorted(exported)


__all__ = ["namespace_attach", "namespace_dir", "namespace_exports", "namespace_getattr"]
