"""Overview of vectorstore faiss.

This module bundles vectorstore faiss logic for the kgfoundry stack. It groups related helpers so
downstream packages can import a single cohesive namespace. Refer to the functions and classes below
for implementation specifics.
"""

from kgfoundry_common.navmap_types import NavMap
from vectorstore_faiss import gpu

__all__ = ["gpu"]

__navmap__: NavMap = {
    "title": "vectorstore_faiss",
    "synopsis": "FAISS GPU vector store wrappers",
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
        "since": "0.2.0",
    },
    "symbols": {
        "gpu": {
            "stability": "experimental",
            "owner": "@search-api",
            "since": "0.2.0",
        },
    },
}

# [nav:anchor gpu]
