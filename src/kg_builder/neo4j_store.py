"""Placeholder interface for a Neo4j-backed store used in demos.

NavMap:
- Neo4jStore: Placeholder interface for a Neo4j-backed store.
"""

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
            "symbols": ["Neo4jStore"],
        },
    ],
}


# [nav:anchor Neo4jStore]
class Neo4jStore:
    """Placeholder interface for a Neo4j-backed store."""

    ...
