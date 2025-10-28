"""Loader utilities."""

from __future__ import annotations

from typing import Final

from kgfoundry_common.navmap_types import NavMap

__all__ = ["OntologyLoader"]

__navmap__: Final[NavMap] = {
    "title": "ontology.loader",
    "synopsis": "Ontology ingest and caching helpers",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@ontology",
        "stability": "experimental",
        "since": "0.1.0",
    },
    "symbols": {
        "OntologyLoader": {
            "owner": "@ontology",
            "stability": "experimental",
            "since": "0.1.0",
        },
    },
}


# [nav:anchor OntologyLoader]
class OntologyLoader:
    """Describe OntologyLoader."""

    ...
