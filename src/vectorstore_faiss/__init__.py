"""Vectorstore Faiss utilities."""

from kgfoundry_common.navmap_types import NavMap
from vectorstore_faiss import gpu

__all__ = ["gpu"]

__navmap__: NavMap = {
    "title": "vectorstore_faiss",
    "synopsis": "FAISS-backed vector store abstractions",
    "exports": __all__,
    "owner": "@vectorstore",
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
        "gpu": {
            "stability": "experimental",
            "owner": "@vectorstore",
            "since": "0.1.0",
        },
    },
}

# [nav:anchor gpu]
