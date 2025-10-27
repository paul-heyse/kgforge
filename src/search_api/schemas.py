"""Module for search_api.schemas."""


from __future__ import annotations
from typing import Optional, Dict, List
from pydantic import BaseModel, Field

class SearchRequest(BaseModel):
    """Searchrequest."""
    query: str = Field(min_length=1)
    k: int = 10
    filters: Optional[Dict[str, object]] = None
    explain: bool = False

class SearchResult(BaseModel):
    """Searchresult."""
    doc_id: str
    chunk_id: str
    title: str
    section: str
    score: float
    signals: Dict[str, float] = {}
    spans: Dict[str, int] = {}
    concepts: List[Dict[str, str]] = []
