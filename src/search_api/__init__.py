"""Overview of search api.

This module bundles search api logic for the kgfoundry stack. It groups
related helpers so downstream packages can import a single cohesive
namespace. Refer to the functions and classes below for implementation
specifics.
"""

# [nav:anchor app]
# [nav:anchor bm25_index]
# [nav:anchor faiss_adapter]
# [nav:anchor fixture_index]
# [nav:anchor fusion]
# [nav:anchor kg_mock]
# [nav:anchor schemas]
# [nav:anchor service]
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
)

# [nav:anchor splade_index]

__all__ = [
    "app",
    "bm25_index",
    "faiss_adapter",
    "fixture_index",
    "fusion",
    "kg_mock",
    "schemas",
    "service",
    "splade_index",
]

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
