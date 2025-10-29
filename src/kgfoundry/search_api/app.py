"""Expose ``search_api.app`` inside the ``kgfoundry`` namespace."""

from __future__ import annotations

from typing import Any

import search_api.app as _module

from search_api.app import (
    apply_kg_boosts,
    app as app,
    auth,
    bm25,
    graph_concepts,
    healthz,
    rrf_fuse,
    search,
)

__all__ = [
    "apply_kg_boosts",
    "app",
    "auth",
    "bm25",
    "graph_concepts",
    "healthz",
    "rrf_fuse",
    "search",
]
__doc__ = _module.__doc__
__path__ = list(getattr(_module, "__path__", []))


def __getattr__(name: str) -> Any:
    return getattr(_module, name)


def __dir__() -> list[str]:
    candidates = set(__all__)
    candidates.update(name for name in dir(_module) if not name.startswith("__"))
    return sorted(candidates)
