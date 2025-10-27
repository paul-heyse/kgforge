"""
Provide utilities for module.

Notes
-----
This module exposes the primary interfaces for the package.

See Also
--------
search_api.schemas
"""


from __future__ import annotations

from typing import Final

from pydantic import BaseModel, Field

from kgfoundry_common.navmap_types import NavMap

__all__ = ["SearchRequest", "SearchResult"]

__navmap__: Final[NavMap] = {
    "title": "search_api.schemas",
    "synopsis": "Module for search_api.schemas",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": ["SearchRequest", "SearchResult"],
        },
    ],
}


# [nav:anchor SearchRequest]
class SearchRequest(BaseModel):
    """
    Represent SearchRequest.
    
    Attributes
    ----------
    query : str
        Attribute description.
    k : int
        Attribute description.
    filters : Mapping[str, object] | None
        Attribute description.
    explain : bool
        Attribute description.
    
    Examples
    --------
    >>> from search_api.schemas import SearchRequest
    >>> result = SearchRequest()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    search_api.schemas
    
    Notes
    -----
    Document class invariants and lifecycle details here.
    """
    
    
    
    

    query: str = Field(min_length=1)
    k: int = 10
    filters: dict[str, object] | None = None
    explain: bool = False


# [nav:anchor SearchResult]
class SearchResult(BaseModel):
    """
    Represent SearchResult.
    
    Attributes
    ----------
    doc_id : str
        Attribute description.
    chunk_id : str
        Attribute description.
    title : str
        Attribute description.
    section : str
        Attribute description.
    score : float
        Attribute description.
    signals : Mapping[str, float]
        Attribute description.
    spans : Mapping[str, int]
        Attribute description.
    concepts : List[dict[str, str]]
        Attribute description.
    
    Examples
    --------
    >>> from search_api.schemas import SearchResult
    >>> result = SearchResult()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    search_api.schemas
    
    Notes
    -----
    Document class invariants and lifecycle details here.
    """
    
    
    
    

    doc_id: str
    chunk_id: str
    title: str
    section: str
    score: float
    signals: dict[str, float] = {}
    spans: dict[str, int] = {}
    concepts: list[dict[str, str]] = []
