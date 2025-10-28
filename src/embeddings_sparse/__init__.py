"""Embeddings Sparse utilities."""

from embeddings_sparse import base, bm25, splade
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
        "base": {
            "stability": "beta",
            "owner": "@embeddings",
            "since": "0.1.0",
        },
        "bm25": {
            "stability": "beta",
            "owner": "@embeddings",
            "since": "0.1.0",
        },
        "splade": {
            "stability": "experimental",
            "owner": "@embeddings",
            "since": "0.2.0",
        },
    },
}

# [nav:anchor base]
# [nav:anchor bm25]
# [nav:anchor splade]
