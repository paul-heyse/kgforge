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
        Module that receives the attribute lookup.
    name : str
        Attribute requested by import machinery or user code.






    Returns
    -------
    object
        Attribute resolved from ``module``.






    Raises
    ------
    AttributeError
        If the attribute is missing on ``module``.
    """
    return getattr(module, name)


def namespace_dir(module: ModuleType, exports: Iterable[str]) -> list[str]:
    """Build a predictable ``dir`` listing for namespace bridges.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    module : ModuleType
        Module that holds the concrete implementation.
    exports : Iterable[str]
        Symbol names explicitly exported by the proxy module.






    Returns
    -------
    list[str]
        Sorted set of attribute names surfaced to callers.
    """
    candidates = set(exports)
    candidates.update(attr for attr in dir(module) if not attr.startswith("__"))
    return sorted(candidates)


__all__ = ["namespace_dir", "namespace_getattr"]
