"""Overview of embeddings sparse.

This module bundles embeddings sparse logic for the kgfoundry stack. It groups related helpers so
downstream packages can import a single cohesive namespace. Refer to the functions and classes below
for implementation specifics.
"""


from embeddings_sparse import base, bm25, splade
from kgfoundry_common.navmap_types import NavMap

__all__ = ["base", "bm25", "splade"]

__navmap__: NavMap = {
    "title": "embeddings_sparse",
    "synopsis": "Sparse embedding adapters and indices",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@embeddings",
        "stability": "experimental",
        "since": "0.1.0",
    },
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
