"""Overview of schemas.

This module bundles schemas logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, Field

from kgfoundry_common.navmap_types import NavMap

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
    """Model the SearchRequest.

    Pydantic model defining the structured payload used across the system. Validation ensures inputs
    conform to the declared schema while producing clear error messages. Use this class when
    serialising or parsing data for the surrounding feature.
    """

    query: str = Field(min_length=1)
    k: int = 10
    filters: dict[str, object] | None = None
    explain: bool = False


# [nav:anchor SearchResult]
class SearchResult(BaseModel):
    """Model the SearchResult.

    Pydantic model defining the structured payload used across the system. Validation ensures inputs
    conform to the declared schema while producing clear error messages. Use this class when
    serialising or parsing data for the surrounding feature.
    """

    doc_id: str
    chunk_id: str
    title: str
    section: str
    score: float
    signals: dict[str, float] = {}
    spans: dict[str, int] = {}
    concepts: list[dict[str, str]] = []
