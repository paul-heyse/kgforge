"""
Provide utilities for module.

Notes
-----
This module exposes the primary interfaces for the package.

See Also
--------
kg_builder.mock_kg
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
    """
    Represent MockKG.
    
    Attributes
    ----------
    None
        No public attributes documented.
    
    Methods
    -------
    __init__()
        Method description.
    add_mention()
        Method description.
    add_edge()
        Method description.
    linked_concepts()
        Method description.
    one_hop()
        Method description.
    
    Examples
    --------
    >>> from kg_builder.mock_kg import MockKG
    >>> result = MockKG()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    kg_builder.mock_kg
    
    Notes
    -----
    Document class invariants and lifecycle details here.
    """
    
    

    def __init__(self) -> None:
        """
        Return init.
        
        Examples
        --------
        >>> from kg_builder.mock_kg import __init__
        >>> __init__()  # doctest: +ELLIPSIS
        
        See Also
        --------
        kg_builder.mock_kg
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        self.chunk2concepts: dict[str, set[str]] = {}
        self.neighbors: dict[str, set[str]] = {}

    def add_mention(self, chunk_id: str, concept_id: str) -> None:
        """
        Return add mention.
        
        Parameters
        ----------
        chunk_id : str
            Description for ``chunk_id``.
        concept_id : str
            Description for ``concept_id``.
        
        Examples
        --------
        >>> from kg_builder.mock_kg import add_mention
        >>> add_mention(..., ...)  # doctest: +ELLIPSIS
        
        See Also
        --------
        kg_builder.mock_kg
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        self.chunk2concepts.setdefault(chunk_id, set()).add(concept_id)

    def add_edge(self, a: str, b: str) -> None:
        """
        Return add edge.
        
        Parameters
        ----------
        a : str
            Description for ``a``.
        b : str
            Description for ``b``.
        
        Examples
        --------
        >>> from kg_builder.mock_kg import add_edge
        >>> add_edge(..., ...)  # doctest: +ELLIPSIS
        
        See Also
        --------
        kg_builder.mock_kg
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        self.neighbors.setdefault(a, set()).add(b)
        self.neighbors.setdefault(b, set()).add(a)

    def linked_concepts(self, chunk_id: str) -> list[str]:
        """
        Return linked concepts.
        
        Parameters
        ----------
        chunk_id : str
            Description for ``chunk_id``.
        
        Returns
        -------
        List[str]
            Description of return value.
        
        Examples
        --------
        >>> from kg_builder.mock_kg import linked_concepts
        >>> result = linked_concepts(...)
        >>> result  # doctest: +ELLIPSIS
        ...
        
        See Also
        --------
        kg_builder.mock_kg
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        
        return sorted(self.chunk2concepts.get(chunk_id, set()))

    def one_hop(self, concept_id: str) -> list[str]:
        """
        Return one hop.
        
        Parameters
        ----------
        concept_id : str
            Description for ``concept_id``.
        
        Returns
        -------
        List[str]
            Description of return value.
        
        Examples
        --------
        >>> from kg_builder.mock_kg import one_hop
        >>> result = one_hop(...)
        >>> result  # doctest: +ELLIPSIS
        ...
        
        See Also
        --------
        kg_builder.mock_kg
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        
        return sorted(self.neighbors.get(concept_id, set()))
