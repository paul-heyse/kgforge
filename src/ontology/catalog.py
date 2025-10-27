"""Module for ontology.catalog.

NavMap:
- OntologyCatalog: Ontologycatalog.
"""

from __future__ import annotations

from kgforge.kgforge_common.models import Concept


class OntologyCatalog:
    """Ontologycatalog."""

    def __init__(self, concepts: list[Concept]):
        """Init.

        Args:
            concepts (List[Concept]): TODO.
        """
        self.by_id = {c.id: c for c in concepts}

    def neighbors(self, concept_id: str, depth: int = 1) -> list[str]:
        """Neighbors.

        Args:
            concept_id (str): TODO.
            depth (int): TODO.

        Returns:
            List[str]: TODO.
        """
        # TODO: return neighbor concept IDs up to depth.
        return []
