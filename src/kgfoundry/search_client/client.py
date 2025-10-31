"""Expose ``search_client.client`` inside the ``kgfoundry`` namespace."""

from __future__ import annotations

import search_client.client as _module
from kgfoundry._namespace_proxy import namespace_dir, namespace_getattr
from search_client.client import KGFoundryClient, RequestsHttp, SupportsHttp, SupportsResponse

__all__ = ["KGFoundryClient", "RequestsHttp", "SupportsHttp", "SupportsResponse"]
__doc__ = _module.__doc__
__path__ = list(getattr(_module, "__path__", []))


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
