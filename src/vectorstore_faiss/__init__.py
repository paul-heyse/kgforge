"""Vectorstore Faiss utilities."""

from typing import Final

import vectorstore_faiss.gpu as gpu

from kgfoundry_common.navmap_types import NavMap

__all__ = ["gpu"]

__navmap__: Final[NavMap] = {
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
        "gpu": {},
    },
}
