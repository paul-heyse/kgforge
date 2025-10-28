"""Kg Builder utilities."""

from kg_builder import mock_kg, neo4j_store
from kgfoundry_common.navmap_types import NavMap

__all__ = ["mock_kg", "neo4j_store"]

__navmap__: NavMap = {
    "title": "kg_builder",
    "synopsis": "Knowledge graph builder components and interfaces",
    "exports": __all__,
    "owner": "@kg-builder",
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
