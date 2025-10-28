"""Overview of kg builder.

This module bundles kg builder logic for the kgfoundry stack. It groups related helpers so
downstream packages can import a single cohesive namespace. Refer to the functions and classes below
for implementation specifics.
"""


from kg_builder import mock_kg, neo4j_store
from kgfoundry_common.navmap_types import NavMap

__all__ = ["mock_kg", "neo4j_store"]

__navmap__: NavMap = {
    "title": "kg_builder",
    "synopsis": "Knowledge graph builder components and interfaces",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@kg-builder",
        "stability": "experimental",
        "since": "0.1.0",
    },
    "symbols": {
        "mock_kg": {
            "stability": "experimental",
            "owner": "@kg-builder",
            "since": "0.1.0",
        },
        "neo4j_store": {
            "stability": "experimental",
            "owner": "@kg-builder",
            "since": "0.1.0",
        },
    },
}

# [nav:anchor mock_kg]
# [nav:anchor neo4j_store]
