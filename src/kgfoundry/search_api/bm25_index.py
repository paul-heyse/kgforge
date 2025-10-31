"""Expose ``search_api.bm25_index`` inside the ``kgfoundry`` namespace."""

from __future__ import annotations

from typing import cast

import search_api.bm25_index as _module
from kgfoundry._namespace_proxy import namespace_attach, namespace_dir, namespace_getattr
from search_api.bm25_index import BM25Doc, BM25Index, toks

__all__ = ["BM25Doc", "BM25Index", "toks"]
_namespace = cast(dict[str, object], globals())
namespace_attach(_module, _namespace, __all__)


def __getattr__(name: str) -> object:
    """Document   getattr  .

    &lt;!-- auto:docstring-builder v1 --&gt;

    Provide a fallback for unknown attribute lookups. This special method integrates the class with Python&#39;s data model so instances behave consistently with the language expectations.

    Parameters
    ----------
    name : str
        Configure the name.


    Returns
    -------
    object
        Describe return value.
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
