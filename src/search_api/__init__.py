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
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from search_api import (
        app,
        bm25_index,
        faiss_adapter,
        fixture_index,
        fusion,
        kg_mock,
        schemas,
        service,
        splade_index,
        types,
    )

__all__: tuple[str, ...] = (
    "app",
    "bm25_index",
    "faiss_adapter",
    "fixture_index",
    "fusion",
    "kg_mock",
    "schemas",
    "service",
    "splade_index",
    "types",
)


def __getattr__(name: str) -> object:
    """Provide lazy module loading for submodules.

    Implements lazy loading for submodules in the search_api package.
    When a submodule is accessed (e.g., `search_api.app`), it is
    dynamically imported and cached in sys.modules.

    Parameters
    ----------
    name : str
        Submodule name to import (e.g., "app", "bm25_index").

    Returns
    -------
    object
        Imported module object.

    Raises
    ------
    AttributeError
        If the requested name is not in the allowed aliases.
    """
    if name not in _ALIASES:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message) from None
    module = importlib.import_module(_ALIASES[name])
    sys.modules[f"{__name__}.{name}"] = module
    return module


def __dir__() -> list[str]:
    """Return list of public module names.

    Returns the list of public submodule names available in this package.
    Used by dir() to show available attributes.

    Returns
    -------
    list[str]
        Sorted list of public module names from __all__.
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
