"""Overview of linker.

This module bundles linker logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
"""


from __future__ import annotations

from typing import Final

from kgfoundry_common.navmap_types import NavMap

__all__ = ["Linker"]

__navmap__: Final[NavMap] = {
    "title": "linking.linker",
    "synopsis": "Linking orchestrations and scoring helpers",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@linking",
        "stability": "experimental",
        "since": "0.1.0",
    },
    "symbols": {
        "Linker": {
            "owner": "@linking",
            "stability": "experimental",
            "since": "0.1.0",
        },
    },
}


# [nav:anchor Linker]
class Linker:
    """Model the Linker.

    Represent the linker data structure used throughout the project. The class encapsulates
    behaviour behind a well-defined interface for collaborating components. Instances are typically
    created by factories or runtime orchestrators documented nearby.
    """
    

    ...
