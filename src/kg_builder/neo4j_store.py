"""Neo4J Store utilities."""

from __future__ import annotations

from typing import Final

from kgfoundry_common.navmap_types import NavMap

__all__ = ["Neo4jStore"]

__navmap__: Final[NavMap] = {
    "title": "kg_builder.neo4j_store",
    "synopsis": "Placeholder interface for a Neo4j-backed store",
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
        "Neo4jStore": {
            "owner": "@kg-builder",
            "stability": "experimental",
            "since": "0.1.0",
        },
    },
}


# [nav:anchor Neo4jStore]
class Neo4jStore:
    """Describe Neo4jStore."""

    ...
