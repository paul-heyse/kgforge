"""Module for kg_builder.mock_kg.

NavMap:
- MockKG: A tiny in-memory KG for demo.
"""

from __future__ import annotations


class MockKG:
    """A tiny in-memory KG for demo.

    Maps chunk_id -> set(concept_id) and concept adjacency.
    """

    def __init__(self) -> None:
        """Init."""
        self.chunk2concepts: dict[str, set[str]] = {}
        self.neighbors: dict[str, set[str]] = {}

    def add_mention(self, chunk_id: str, concept_id: str) -> None:
        """Add a mention linking a chunk to a concept.

        Parameters
        ----------
        chunk_id : str
            TODO.
        concept_id : str
            TODO.
        """
        self.chunk2concepts.setdefault(chunk_id, set()).add(concept_id)

    def add_edge(self, a: str, b: str) -> None:
        """Add an undirected edge between two concepts.

        Parameters
        ----------
        a : str
            TODO.
        b : str
            TODO.
        """
        self.neighbors.setdefault(a, set()).add(b)
        self.neighbors.setdefault(b, set()).add(a)

    def linked_concepts(self, chunk_id: str) -> list[str]:
        """Return the concepts linked to a chunk identifier.

        Parameters
        ----------
        chunk_id : str
            TODO.

        Returns
        -------
        list[str]
            TODO.
        """
        return sorted(self.chunk2concepts.get(chunk_id, set()))

    def one_hop(self, concept_id: str) -> list[str]:
        """Return one-hop neighbours for a concept identifier.

        Parameters
        ----------
        concept_id : str
            TODO.

        Returns
        -------
        list[str]
            TODO.
        """
        return sorted(self.neighbors.get(concept_id, set()))
