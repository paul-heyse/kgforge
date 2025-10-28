"""Search Api utilities."""

from typing import Final

import search_api.app as app
import search_api.bm25_index as bm25_index
import search_api.faiss_adapter as faiss_adapter
import search_api.fixture_index as fixture_index
import search_api.fusion as fusion
import search_api.kg_mock as kg_mock
import search_api.schemas as schemas
import search_api.service as service
import search_api.splade_index as splade_index

from kgfoundry_common.navmap_types import NavMap

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

__navmap__: Final[NavMap] = {
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
