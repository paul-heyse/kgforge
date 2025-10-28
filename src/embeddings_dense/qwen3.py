"""Qwen3 utilities."""

from __future__ import annotations

from typing import Final

from kgfoundry_common.navmap_types import NavMap

__all__ = ["Qwen3Embedder"]

__navmap__: Final[NavMap] = {
    "title": "embeddings_dense.qwen3",
    "synopsis": "Qwen-3 dense embedding adapter",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": ["Qwen3Embedder"],
        },
    ],
}


# [nav:anchor Qwen3Embedder]
class Qwen3Embedder:
    """Describe Qwen3Embedder."""

    ...
