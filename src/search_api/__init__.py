"""Search service endpoints and retrieval adapters.

This package provides typed search APIs with schema validation and
Problem Details error responses. All public APIs are explicitly exported
via `__all__` with full type annotations.

See Also
--------
- `schema/examples/problem_details/search-missing-index.json` - Example error response
- `schema/models/search_request.v1.json` - Request schema
- `schema/models/search_result.v1.json` - Response schema
"""
# [nav:section public-api]

from __future__ import annotations

import sys
from importlib import import_module
from types import ModuleType

from kgfoundry_common.navmap_loader import load_nav_metadata

_ALIASES: dict[str, str] = {
    "app": "search_api.app",
    "bm25_index": "search_api.bm25_index",
    "faiss_adapter": "search_api.faiss_adapter",
    "fixture_index": "search_api.fixture_index",
    "fusion": "search_api.fusion",
    "kg_mock": "search_api.kg_mock",
    "schemas": "search_api.schemas",
    "service": "search_api.service",
    "splade_index": "search_api.splade_index",
    "types": "search_api.types",
}

__all__ = list(_ALIASES)

__navmap__ = load_nav_metadata(__name__, tuple(__all__))


def _load(name: str) -> ModuleType:
    module = import_module(_ALIASES[name])
    sys.modules[f"{__name__}.{name}"] = module
    return module


def __getattr__(name: str) -> ModuleType:
    if name not in _ALIASES:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    return _load(name)


def __dir__() -> list[str]:
    return sorted(set(__all__))
