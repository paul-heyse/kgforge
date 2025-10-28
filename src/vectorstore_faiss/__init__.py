"""Vectorstore Faiss utilities."""

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
