"""Ontology utilities."""

# [nav:anchor catalog]
# [nav:anchor loader]
from kgfoundry_common.navmap_types import NavMap
from ontology import catalog, loader

__all__ = ["catalog", "loader"]

__navmap__: NavMap = {
    "title": "ontology",
    "synopsis": "Ontology loading and lookup helpers",
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
