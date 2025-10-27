"""Utilities for loading ontology catalog data into the registry."""

from __future__ import annotations

from typing import Final

from kgfoundry_common.navmap_types import NavMap

__all__ = ["OntologyLoader"]

__navmap__: Final[NavMap] = {
    "title": "ontology.loader",
    "synopsis": "Module for ontology.loader",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": ["OntologyLoader"],
        },
    ],
}


# [nav:anchor OntologyLoader]
class OntologyLoader:
    """Placeholder loader for ontology metadata."""

    ...
