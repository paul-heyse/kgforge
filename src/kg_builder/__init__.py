"""Kg Builder utilities."""

from typing import Final

import kg_builder.mock_kg as mock_kg
import kg_builder.neo4j_store as neo4j_store

from kgfoundry_common.navmap_types import NavMap

__all__ = ["mock_kg", "neo4j_store"]

__navmap__: Final[NavMap] = {
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
        "mock_kg": {},
        "neo4j_store": {},
    },
}
