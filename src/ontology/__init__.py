"""Overview of ontology.

This module bundles ontology logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
"""


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
