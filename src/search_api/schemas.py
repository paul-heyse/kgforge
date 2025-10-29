"""Overview of schemas.

This module bundles schemas logic for the kgfoundry stack. It groups
related helpers so downstream packages can import a single cohesive
namespace. Refer to the functions and classes below for implementation
specifics.
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
    """Describe SearchRequest.

<!-- auto:docstring-builder v1 -->

the behaviour it provides to callers. Callers interact with
validated data through this model.

Parameters
----------
query : str
    Describe ``query``.
k : int, optional
    Describe ``k``.
    Defaults to ``10``.
filters : dict[str, object] | None, optional
    Describe ``filters``.
    Defaults to ``None``.
explain : bool, optional
    Describe ``explain``.
    Defaults to ``False``.
"""

    query: str = Field(min_length=1)
    k: int = 10
    filters: dict[str, object] | None = None
    explain: bool = False


# [nav:anchor SearchResult]
class SearchResult(BaseModel):
    """Describe SearchResult.

<!-- auto:docstring-builder v1 -->

the behaviour it provides to callers. Callers interact with
validated data through this model.

Parameters
----------
doc_id : str
    Describe ``doc_id``.
chunk_id : str
    Describe ``chunk_id``.
title : str
    Describe ``title``.
section : str
    Describe ``section``.
score : float
    Describe ``score``.
signals : dict[str, float], optional
    Describe ``signals``.
    Defaults to ``<factory>``.
spans : dict[str, int], optional
    Describe ``spans``.
    Defaults to ``<factory>``.
concepts : list[dict[str, str]], optional
    Describe ``concepts``.
    Defaults to ``<factory>``.
"""

    doc_id: str
    chunk_id: str
    title: str
    section: str
    score: float
    signals: dict[str, float] = Field(default_factory=dict)
    spans: dict[str, int] = Field(default_factory=dict)
    concepts: list[dict[str, str]] = Field(default_factory=list)
