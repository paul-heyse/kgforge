"""Module for search_api.schemas.

NavMap:
- SearchRequest: Searchrequest.
- SearchResult: Searchresult.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Searchrequest."""

    query: str = Field(min_length=1)
    k: int = 10
    filters: dict[str, object] | None = None
    explain: bool = False


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
