"""Search Api utilities."""

from kgfoundry_common.navmap_types import NavMap
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

__navmap__: NavMap = {
    "title": "search_api",
    "synopsis": "Search service endpoints and retrieval adapters",
    "exports": __all__,
    "owner": "@search-api",
    "stability": "experimental",
    "since": "0.1.0",
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "symbols": {name: {} for name in __all__},
}
