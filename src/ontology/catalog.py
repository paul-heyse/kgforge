"""Catalog utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from kgfoundry_common.navmap_types import NavMap


@dataclass
class Concept:
    """Lightweight concept record used for typing within the ontology layer."""

    id: str
    label: str | None = None


__all__ = ["OntologyCatalog"]

__navmap__: Final[NavMap] = {
    "title": "ontology.catalog",
    "synopsis": "Utility catalogue for lightweight ontology lookups.",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@ontology",
        "stability": "beta",
        "since": "0.1.0",
    },
    "symbols": {
        "OntologyCatalog": {
            "owner": "@ontology",
            "stability": "beta",
            "since": "0.1.0",
        },
    },
}


# [nav:anchor OntologyCatalog]
class OntologyCatalog:
    """Describe OntologyCatalog."""

    def __init__(self, concepts: list[Concept]) -> None:
        """Compute init.

        Initialise a new instance with validated parameters.

        Parameters
        ----------
        concepts : List[src.ontology.catalog.Concept]
            Description for ``concepts``.
        """
        self.by_id = {concept.id: concept for concept in concepts}

    def neighbors(self, concept_id: str, depth: int = 1) -> set[str]:
        """Compute neighbors.

        Carry out the neighbors operation.

        Parameters
        ----------
        concept_id : str
            Description for ``concept_id``.
        depth : int | None
            Description for ``depth``.

        Returns
        -------
        collections.abc.Set
            Description of return value.

        Examples
        --------
        >>> from ontology.catalog import neighbors
        >>> result = neighbors(..., ...)
        >>> result  # doctest: +ELLIPSIS
        ...
        """
        # NOTE: return neighbor concept IDs up to depth when ontology data is wired
        return set()

    def hydrate(self, concept_id: str) -> dict[str, Any]:
        """Compute hydrate.

        Carry out the hydrate operation.

        Parameters
        ----------
        concept_id : str
            Description for ``concept_id``.

        Returns
        -------
        collections.abc.Mapping
            Description of return value.

        Examples
        --------
        >>> from ontology.catalog import hydrate
        >>> result = hydrate(...)
        >>> result  # doctest: +ELLIPSIS
        ...
        """
        return {}
