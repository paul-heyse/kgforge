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

from __future__ import annotations

import importlib
import sys

# [nav:anchor app]
# [nav:anchor bm25_index]
# [nav:anchor faiss_adapter]
# [nav:anchor fixture_index]
# [nav:anchor fusion]
# [nav:anchor kg_mock]
# [nav:anchor schemas]
# [nav:anchor service]

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

__all__ = sorted(_ALIASES)


def __getattr__(name: str) -> object:
    """Describe   getattr  .

    <!-- auto:docstring-builder v1 -->

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

    Raises
    ------
    AttributeError
        Raised when message.
"""
    if name not in _ALIASES:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message) from None
    module = importlib.import_module(_ALIASES[name])
    sys.modules[f"{__name__}.{name}"] = module
    return module


def __dir__() -> list[str]:
    """Describe   dir  .

    <!-- auto:docstring-builder v1 -->

    &lt;!-- auto:docstring-builder v1 --&gt;

    Expose the attributes reported when ``dir()`` is called on the instance. This special method integrates the class with Python&#39;s data model so instances behave consistently with the language expectations.

    Returns
    -------
    list[str]
        Describe return value.
"""
    return sorted(set(__all__))


__navmap__ = {
    "title": "search_api",
    "synopsis": "Search service endpoints and retrieval adapters",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@search-api",
        "stability": "experimental",
        "since": "0.1.0",
    },
    "symbols": {
        name: {
            "owner": "@search-api",
            "stability": "experimental",
            "since": "0.1.0",
        }
        for name in __all__
    },
}
