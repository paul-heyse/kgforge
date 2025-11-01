"""Expose ``search_client.client`` inside the ``kgfoundry`` namespace."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import search_client.client as _module
from kgfoundry.tooling_bridge import namespace_dir, namespace_getattr
from search_client.client import KGFoundryClient, RequestsHttp, SupportsHttp, SupportsResponse

__all__ = ["KGFoundryClient", "RequestsHttp", "SupportsHttp", "SupportsResponse"]
_module_doc = cast(str | None, getattr(_module, "__doc__", None))
if isinstance(_module_doc, str):  # pragma: no branch - simple type guard
    __doc__ = _module_doc

_module_path = cast(Sequence[object] | None, getattr(_module, "__path__", None))
__path__ = [str(item) for item in _module_path] if isinstance(_module_path, Sequence) else []


def __getattr__(name: str) -> object:
    """Return ``name`` from the proxied ``search_client.client`` module."""
    return namespace_getattr(_module, name)


def __dir__() -> list[str]:
    """Return the combined attribute listing exposed by the namespace bridge."""
    return namespace_dir(_module, __all__)
