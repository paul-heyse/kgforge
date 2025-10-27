"""Module for docling.vlm.

NavMap:
- NavMap: Structure describing a module navmap.
- GraniteDoclingVLM: Expose unified utilities for Docling-compatible VLMs.
"""

from __future__ import annotations

from typing import Final

from kgfoundry_common.navmap_types import NavMap

__all__ = ["GraniteDoclingVLM"]

__navmap__: Final[NavMap] = {
    "title": "docling.vlm",
    "synopsis": "Vision-language tagging helpers for docling",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": ["GraniteDoclingVLM"],
        },
    ],
}


# [nav:anchor GraniteDoclingVLM]
class GraniteDoclingVLM:
    """Expose unified utilities for Docling-compatible VLMs."""

    ...
