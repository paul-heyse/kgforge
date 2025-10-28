"""Docling utilities."""

from docling import canonicalizer, hybrid, vlm
from kgfoundry_common.navmap_types import NavMap

__all__ = ["canonicalizer", "hybrid", "vlm"]

__navmap__: NavMap = {
    "title": "docling",
    "synopsis": "Public surface for docling preprocessing utilities",
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
        "stability": "experimental",
        "since": "0.1.0",
    },
    "symbols": {
        "canonicalizer": {
            "stability": "beta",
            "owner": "@docling",
            "since": "0.1.0",
        },
        "hybrid": {
            "stability": "beta",
            "owner": "@docling",
            "since": "0.1.0",
        },
        "vlm": {
            "stability": "experimental",
            "owner": "@docling",
            "since": "0.1.0",
        },
    },
}

# [nav:anchor canonicalizer]
# [nav:anchor hybrid]
# [nav:anchor vlm]
