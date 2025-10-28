"""Overview of hybrid.

This module bundles hybrid logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
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
    """Model the HybridChunker.

    Represent the hybridchunker data structure used throughout the project. The class encapsulates
    behaviour behind a well-defined interface for collaborating components. Instances are typically
    created by factories or runtime orchestrators documented nearby.
    """

    ...
