"""Overview of schemas.

This module bundles schemas logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
"""

from __future__ import annotations

from typing import Final

from pydantic import Field

from kgfoundry_common.navmap_types import NavMap
from kgfoundry_common.pydantic import BaseModel

__all__ = ["SearchRequest", "SearchResult"]

__navmap__: Final[NavMap] = {
    "title": "search_api.schemas",
    "synopsis": "Pydantic models used by the search API",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@search-api",
        "stability": "experimental",
        "since": "0.2.0",
    },
    "symbols": {
        name: {
            "owner": "@search-api",
            "stability": "experimental",
            "since": "0.2.0",
        }
        for name in __all__
    },
}


# [nav:anchor SearchRequest]
class SearchRequest(BaseModel):
    """Request model describing an incoming search query.
<!-- auto:docstring-builder v1 -->

    Attributes
    ----------
    query : str
        Full-text query string supplied by the caller.
    k : int
        Maximum number of results to return. Defaults to ``10``.
    filters : dict[str, object] | None
        Optional filter expression forwarded to the backend.
    explain : bool
        Flag indicating whether to return scoring breakdowns. Defaults to ``False``.
    """

    query: str = Field(min_length=1)
    k: int = 10
    filters: dict[str, object] | None = None
    explain: bool = False


# [nav:anchor SearchResult]
class SearchResult(BaseModel):
    """Result model produced by the search API.
<!-- auto:docstring-builder v1 -->

    Attributes
    ----------
    doc_id : str
        Identifier for the matched document.
    chunk_id : str
        Identifier for the specific chunk within the document.
    title : str
        Title of the source document.
    section : str
        Section heading captured for the chunk.
    score : float
        Score assigned by the retrieval backend.
    signals : dict[str, float]
        Optional per-feature contribution breakdown.
    spans : dict[str, int]
        Mapping of matched spans to offsets.
    concepts : list[dict[str, str]]
        Optional knowledge graph annotations returned with the result.
    """

    doc_id: str
    chunk_id: str
    title: str
    section: str
    score: float
    signals: dict[str, float] = Field(default_factory=dict)
    spans: dict[str, int] = Field(default_factory=dict)
    concepts: list[dict[str, str]] = Field(default_factory=list)
