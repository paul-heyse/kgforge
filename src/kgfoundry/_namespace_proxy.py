"""Helpers for namespace bridge modules."""

from __future__ import annotations

from collections.abc import Iterable
from types import ModuleType


def namespace_getattr(module: ModuleType, name: str) -> object:
    """Fetch a proxied attribute from the underlying module.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    module : ModuleType
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


def namespace_dir(module: ModuleType, exports: Iterable[str]) -> list[str]:
    """Build a predictable ``dir`` listing for namespace bridges.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    module : ModuleType
        Describe ``module``.
    exports : Iterable[str]
        Describe ``exports``.


    Returns
    -------
    list[str]
        Sorted set of attribute names surfaced to callers.
    """
    candidates = set(exports)
    candidates.update(attr for attr in dir(module) if not attr.startswith("__"))
    return sorted(candidates)


__all__ = ["namespace_dir", "namespace_getattr"]
