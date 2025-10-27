"""Module for docling.canonicalizer.

NavMap:
- NavMap: Structure describing a module navmap.
- canonicalize_text: Apply NFC, normalize whitespace and bullets, preserve….
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

from kgfoundry_common.navmap_types import NavMap

__all__ = ["canonicalize_text"]

__navmap__: Final[NavMap] = {
    "title": "docling.canonicalizer",
    "synopsis": "String canonicalisation utilities for docling preprocessing",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": ["canonicalize_text"],
        },
    ],
}


# [nav:anchor canonicalize_text]
def canonicalize_text(blocks: list[str]) -> str:
    """Apply NFC, normalize whitespace and bullets, preserve single newlines between blocks."""

    def norm(s: str) -> str:
        """Normalise a single text block to a canonical representation."""
        s = unicodedata.normalize("NFC", s)
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        s = re.sub(r"[\u2022\u25E6\u2013]", "-", s)  # bullets/dashes
        s = re.sub(r"[\x00-\x1F]", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    normed = [norm(b) for b in blocks if b.strip()]
    return "\n".join(normed)
