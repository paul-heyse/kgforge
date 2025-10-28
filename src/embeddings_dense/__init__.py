"""Embeddings Dense utilities."""

import embeddings_dense.base as base
import embeddings_dense.qwen3 as qwen3

from kgfoundry_common.navmap_types import NavMap

__all__ = ["base", "qwen3"]

__navmap__: NavMap = {
    "title": "embeddings_dense",
    "synopsis": "Dense embedding adapters and protocols",
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
        "qwen3": {},
    },
}
