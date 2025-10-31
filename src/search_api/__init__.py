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
    types,
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
    "types",
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
