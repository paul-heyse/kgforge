"""Expose ``search_api.faiss_adapter`` inside the ``kgfoundry`` namespace."""

from __future__ import annotations

import search_api.faiss_adapter as _module
from kgfoundry._namespace_proxy import namespace_dir, namespace_getattr
from search_api.faiss_adapter import HAVE_FAISS, DenseVecs, FaissAdapter, VecArray

__all__ = ["HAVE_FAISS", "DenseVecs", "FaissAdapter", "VecArray"]
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
