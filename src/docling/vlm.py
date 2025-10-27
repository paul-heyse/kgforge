"""Vision-language model helpers used by docling for page-level tagging.

NavMap:
- generate_page_tags: Produce VLM tags for document pages.
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
