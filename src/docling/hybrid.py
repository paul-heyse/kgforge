"""Module for docling.hybrid.

NavMap:
- NavMap: Structure describing a module navmap.
- HybridChunker: Placeholder hybrid chunker for Docling outputs.
"""

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
            "symbols": ["HybridChunker"],
        },
    ],
}


# [nav:anchor HybridChunker]
class HybridChunker:
    """Placeholder hybrid chunker for Docling outputs."""

    ...
