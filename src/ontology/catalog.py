"""Module for ontology.catalog.

NavMap:
- OntologyCatalog: Ontologycatalog.
"""

from __future__ import annotations

from kgforge.kgforge_common.models import Concept


class OntologyCatalog:
    """Ontologycatalog."""

    def __init__(self, concepts: list[Concept]) -> None:
        """Init.

        Parameters
        ----------
        concepts : List[Concept]
            TODO.
        """
        self.by_id = {c.id: c for c in concepts}

    def neighbors(self, concept_id: str, depth: int = 1) -> list[str]:
        """Neighbors.

        Parameters
        ----------
        concept_id : str
            TODO.
        depth : int
            TODO.

        Returns
        -------
        List[str]
            TODO.
        """
        # NOTE: return neighbor concept IDs up to depth when ontology data is wired
        return []
