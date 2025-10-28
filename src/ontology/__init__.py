"""Ontology utilities."""

from kgfoundry_common.navmap_types import NavMap
from ontology import catalog, loader

__all__ = ["catalog", "loader"]

__navmap__: NavMap = {
    "title": "ontology",
    "synopsis": "Ontology loading and lookup helpers",
    "exports": __all__,
    "owner": "@ontology",
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
        "catalog": {
            "stability": "beta",
            "owner": "@ontology",
            "since": "0.1.0",
        },
        "loader": {
            "stability": "beta",
            "owner": "@ontology",
            "since": "0.1.0",
        },
    },
}

# [nav:anchor catalog]
# [nav:anchor loader]
