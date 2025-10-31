"""Helpers for namespace bridge modules."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Iterable as TypingIterable
from types import ModuleType
from typing import cast


def namespace_getattr(module: ModuleType, name: str) -> object:
    """Fetch a proxied attribute from the underlying module.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    module : module
        Describe ``module``.
    name : str
        Describe ``name``.

    Returns
    -------
    object
        Attribute resolved from ``module``.

    Raises
    ------
    AttributeError
    If the attribute is missing on ``module``.
    """
    # getattr returns object which may contain Any - this is inherent to dynamic attribute access
    return getattr(module, name)  # type: ignore[misc]


def namespace_exports(module: ModuleType) -> list[str]:
    """Return a typed list of public exports for ``module``."""
    exports: object = getattr(module, "__all__", None)
    if isinstance(exports, (list, tuple, set)):
        return [str(item) for item in exports]
    return [name for name in dir(module) if not name.startswith("_")]


def namespace_attach(module: ModuleType, target: dict[str, object], names: TypingIterable[str]) -> None:
    """Populate ``target`` with attributes from ``module`` while preserving typing."""
    for name in names:
        target[name] = cast(object, getattr(module, name))


def namespace_dir(module: ModuleType, exports: Iterable[str]) -> list[str]:
    """Build a predictable ``dir`` listing for namespace bridges.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    module : module
        Describe ``module``.
    exports : str
        Describe ``exports``.

    Returns
    -------
    list[str]
        Sorted set of attribute names surfaced to callers.
    """
    candidates = set(exports)
    candidates.update(attr for attr in dir(module) if not attr.startswith("__"))
    return sorted(candidates)


__all__ = ["namespace_attach", "namespace_dir", "namespace_exports", "namespace_getattr"]
