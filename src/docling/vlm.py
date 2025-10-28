"""Overview of vlm.

This module bundles vlm logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
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
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@docling",
        "stability": "experimental",
        "since": "0.1.0",
    },
    "symbols": {
        "GraniteDoclingVLM": {
            "owner": "@docling",
            "stability": "experimental",
            "since": "0.1.0",
        },
    },
}


# [nav:anchor GraniteDoclingVLM]
class GraniteDoclingVLM:
    """Model the GraniteDoclingVLM.

    Represent the granitedoclingvlm data structure used throughout the project. The class
    encapsulates behaviour behind a well-defined interface for collaborating components. Instances
    are typically created by factories or runtime orchestrators documented nearby.
    """

    ...
