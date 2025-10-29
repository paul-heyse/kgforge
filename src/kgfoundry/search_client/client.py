"""Expose ``search_client.client`` inside the ``kgfoundry`` namespace."""

from __future__ import annotations

from typing import Any

import search_client.client as _module
from search_client.client import KGFoundryClient, RequestsHttp, SupportsHttp, SupportsResponse

__all__ = ["KGFoundryClient", "RequestsHttp", "SupportsHttp", "SupportsResponse"]
__doc__ = _module.__doc__
__path__ = list(getattr(_module, "__path__", []))


def __getattr__(name: str) -> Any:
    """Return an attribute from the underlying module.

    Parameters
    ----------
    name : str
        Public alias exported by the namespace bridge.

    Returns
    -------
    Any
        Module or attribute proxied from the real package.

    Raises
    ------
    AttributeError
        If the alias is not registered.
    """
    return getattr(_module, name)


def __dir__() -> list[str]:
    """Return the combined attribute listing.

    Returns
    -------
    list[str]
        Sorted collection merging bridge-local and implementation attributes.
    """
    candidates = set(__all__)
    candidates.update(name for name in dir(_module) if not name.startswith("__"))
    return sorted(candidates)
