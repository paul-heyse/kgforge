"""Expose ``search_client.client`` inside the ``kgfoundry`` namespace."""

from __future__ import annotations

from typing import Any

import search_client.client as _module

from search_client.client import KGFoundryClient, RequestsHttp, SupportsHttp, SupportsResponse

__all__ = ["KGFoundryClient", "RequestsHttp", "SupportsHttp", "SupportsResponse"]
__doc__ = _module.__doc__
__path__ = list(getattr(_module, "__path__", []))


def __getattr__(name: str) -> Any:
    return getattr(_module, name)


def __dir__() -> list[str]:
    candidates = set(__all__)
    candidates.update(name for name in dir(_module) if not name.startswith("__"))
    return sorted(candidates)
