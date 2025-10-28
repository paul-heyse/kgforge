"""Embeddings Sparse utilities."""

import embeddings_sparse.base as base
import embeddings_sparse.bm25 as bm25
import embeddings_sparse.splade as splade

from kgfoundry_common.navmap_types import NavMap

__all__ = ["base", "bm25", "splade"]

__navmap__: NavMap = {
    "title": "embeddings_sparse",
    "synopsis": "Sparse embedding adapters and indices",
    "exports": __all__,
    "owner": "@embeddings",
    "stability": "experimental",
    "since": "0.1.0",
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "symbols": {
        "base": {},
        "bm25": {},
        "splade": {},
    },
}
