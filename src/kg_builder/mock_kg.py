"""Mock Kg utilities."""

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
    """Describe MockKG."""

    def __init__(self) -> None:
        """Compute init.

        Initialise a new instance with validated parameters.
        """
        
        
        self.chunk2concepts: dict[str, set[str]] = {}
        self.neighbors: dict[str, set[str]] = {}

    def add_mention(self, chunk_id: str, concept_id: str) -> None:
        """Compute add mention.

        Carry out the add mention operation.

        Parameters
        ----------
        chunk_id : str
            Description for ``chunk_id``.
        concept_id : str
            Description for ``concept_id``.
        """
        
        
        self.chunk2concepts.setdefault(chunk_id, set()).add(concept_id)

    def add_edge(self, a: str, b: str) -> None:
        """Compute add edge.

        Carry out the add edge operation.

        Parameters
        ----------
        a : str
            Description for ``a``.
        b : str
            Description for ``b``.
        """
        
        
        self.neighbors.setdefault(a, set()).add(b)
        self.neighbors.setdefault(b, set()).add(a)

    def linked_concepts(self, chunk_id: str) -> list[str]:
        """Compute linked concepts.

        Carry out the linked concepts operation.

        Parameters
        ----------
        chunk_id : str
            Description for ``chunk_id``.

        Returns
        -------
        List[str]
            Description of return value.
        """
        
        
        return sorted(self.chunk2concepts.get(chunk_id, set()))

    def one_hop(self, concept_id: str) -> list[str]:
        """Compute one hop.

        Carry out the one hop operation.

        Parameters
        ----------
        concept_id : str
            Description for ``concept_id``.

        Returns
        -------
        List[str]
            Description of return value.
        """
        
        
        return sorted(self.neighbors.get(concept_id, set()))
