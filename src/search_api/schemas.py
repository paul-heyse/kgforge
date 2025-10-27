"""Module for search_api.schemas.

NavMap:
- NavMap: Structure describing a module navmap.
- SearchRequest: Searchrequest.
- SearchResult: Searchresult.
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
    """Searchrequest."""

    query: str = Field(min_length=1)
    k: int = 10
    filters: dict[str, object] | None = None
    explain: bool = False


# [nav:anchor SearchResult]
class SearchResult(BaseModel):
    """Searchresult."""

    doc_id: str
    chunk_id: str
    title: str
    section: str
    score: float
    signals: dict[str, float] = {}
    spans: dict[str, int] = {}
    concepts: list[dict[str, str]] = []
