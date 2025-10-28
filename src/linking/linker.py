"""Linker utilities."""

from __future__ import annotations

from typing import Final

from kgfoundry_common.navmap_types import NavMap

__all__ = ["Linker"]

__navmap__: Final[NavMap] = {
    "title": "linking.linker",
    "synopsis": "Module for linking.linker",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": ["Linker"],
        },
    ],
}


# [nav:anchor Linker]
class Linker:
    """Describe Linker."""

    ...
