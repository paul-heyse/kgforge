"""Helpers for the MockKG in-memory knowledge graph used in demos.

NavMap:
- MockKG: Tiny graph leveraged by fixtures and walkthroughs.
"""

from __future__ import annotations

from typing import Final

from kgfoundry_common.navmap_types import NavMap

__all__ = ["MockKG"]

__navmap__: Final[NavMap] = {
    "title": "kg_builder.mock_kg",
    "synopsis": "Helpers for the MockKG in-memory knowledge graph",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": ["MockKG"],
        },
    ],
}


# [nav:anchor MockKG]
class MockKG:
    """A tiny in-memory KG for demo.

    Maps chunk_id -> set(concept_id) and concept adjacency.
    """

    def __init__(self) -> None:
        """Init."""
        self.chunk2concepts: dict[str, set[str]] = {}
        self.neighbors: dict[str, set[str]] = {}

    def add_mention(self, chunk_id: str, concept_id: str) -> None:
        """Link a chunk identifier to the provided concept identifier."""
        self.chunk2concepts.setdefault(chunk_id, set()).add(concept_id)

    def add_edge(self, a: str, b: str) -> None:
        """Connect two concept identifiers in both directions."""
        self.neighbors.setdefault(a, set()).add(b)
        self.neighbors.setdefault(b, set()).add(a)

    def linked_concepts(self, chunk_id: str) -> list[str]:
        """Return concepts associated with ``chunk_id`` in sorted order."""
        return sorted(self.chunk2concepts.get(chunk_id, set()))

    def one_hop(self, concept_id: str) -> list[str]:
        """Return sorted concept identifiers directly connected to ``concept_id``."""
        return sorted(self.neighbors.get(concept_id, set()))
