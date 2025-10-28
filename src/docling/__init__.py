"""Docling utilities."""

import docling.canonicalizer as canonicalizer
import docling.hybrid as hybrid
import docling.vlm as vlm

from kgfoundry_common.navmap_types import NavMap

__all__ = ["canonicalizer", "hybrid", "vlm"]

__navmap__: NavMap = {
    "title": "docling",
    "synopsis": "Public surface for docling preprocessing utilities",
    "exports": __all__,
    "owner": "@docling",
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
        "canonicalizer": {},
        "hybrid": {},
        "vlm": {},
    },
}
