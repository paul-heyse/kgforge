"""Expose ``search_api.app`` inside the ``kgfoundry`` namespace."""

from __future__ import annotations

import search_api.app as _module
from kgfoundry._namespace_proxy import namespace_dir, namespace_getattr
from search_api.app import (
    app as app,
)
from search_api.app import (
    apply_kg_boosts,
    auth,
    bm25,
    graph_concepts,
    healthz,
    rrf_fuse,
    search,
)

__all__ = [
    "app",
    "apply_kg_boosts",
    "auth",
    "bm25",
    "graph_concepts",
    "healthz",
    "rrf_fuse",
    "search",
]
__doc__ = _module.__doc__
__path__ = list(getattr(_module, "__path__", []))


def __getattr__(name: str) -> object:
    """Document   getattr  .

    <!-- auto:docstring-builder v1 -->

    Provide a fallback for unknown attribute lookups. This special method integrates the class with Python's data model so instances behave consistently with the language expectations.

    Parameters
    ----------
    name : str
        TODO: describe ``name``.


    Returns
    -------
    object
        TODO: describe return value.
    """
    return namespace_getattr(_module, name)


def __dir__() -> list[str]:
    """Return the combined attribute listing.

    <!-- auto:docstring-builder v1 -->

    Returns
    -------
    inspect._empty
        Sorted union of exports and implementation attributes.
    """
    return namespace_dir(_module, __all__)
