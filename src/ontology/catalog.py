"""Module for ontology.catalog.

NavMap:
- NavMap: Structure describing a module navmap.
- OntologyCatalog: In-memory view of ontology concepts and relationships.
"""

from __future__ import annotations

from typing import Any, Final

from kgfoundry.kgfoundry_common.models import Concept

from kgfoundry_common.navmap_types import NavMap

__all__ = ["OntologyCatalog"]

__navmap__: Final[NavMap] = {
    "title": "ontology.catalog",
    "synopsis": "Utility catalogue for lightweight ontology lookups.",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": ["OntologyCatalog"],
        },
    ],
}


# [nav:anchor OntologyCatalog]
class OntologyCatalog:
    """In-memory view of ontology concepts and relationships."""

    def __init__(self, concepts: list[Concept]) -> None:
        """Index concepts by identifier for rapid lookup.

        Parameters
        ----------
        concepts : list[Concept]
            Concepts loaded from the backing ontology source.
        """
        self.by_id = {concept.id: concept for concept in concepts}

    def neighbors(self, concept_id: str, depth: int = 1) -> set[str]:
        """Return neighbour concept identifiers up to ``depth`` hops."""
        # NOTE: return neighbor concept IDs up to depth when ontology data is wired
        return set()

    def hydrate(self, concept_id: str) -> dict[str, Any]:
        """Load the full concept record for ``concept_id``."""
        return {}
