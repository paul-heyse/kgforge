"""Hybrid utilities."""

from __future__ import annotations

from typing import Final

from kgfoundry_common.navmap_types import NavMap

__all__ = ["HybridChunker"]

__navmap__: Final[NavMap] = {
    "title": "docling.hybrid",
    "synopsis": "Hybrid docling pipeline combining layout and text cues",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@docling",
        "stability": "beta",
        "since": "0.1.0",
    },
    "symbols": {
        "HybridChunker": {
            "owner": "@docling",
            "stability": "beta",
            "since": "0.1.0",
        },
    },
}


# [nav:anchor HybridChunker]
class HybridChunker:
    """Describe HybridChunker."""

    ...
