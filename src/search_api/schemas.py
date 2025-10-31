"""Overview of schemas.

This module bundles schemas logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
"""

from __future__ import annotations

from typing import Final

from pydantic import ConfigDict, Field

from kgfoundry_common.navmap_types import NavMap
from kgfoundry_common.problem_details import JsonValue
from kgfoundry_common.pydantic import BaseModel

__all__ = ["SearchRequest", "SearchResponse", "SearchResult"]

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


# [nav:anchor SearchResponse]
class SearchResponse(BaseModel):
    """Search API response containing results.

    <!-- auto:docstring-builder v1 -->

    Response envelope for search API endpoints.

    Parameters
    ----------
    **data : Any
        Describe ``data``.
    """

    model_config = ConfigDict(extra="forbid")

    results: list[SearchResult] = Field(default_factory=list)


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
    filters : dict[str, object] | NoneType, optional
        Describe ``filters``.
        Defaults to ``None``.
    explain : bool, optional
        Describe ``explain``.
        Defaults to ``False``.

    Examples
    --------
    >>> from pathlib import Path
    >>> from kgfoundry_common.schema_helpers import assert_model_roundtrip
    >>> example_path = (
    ...     Path(__file__).parent.parent.parent
    ...     / "schema"
    ...     / "examples"
    ...     / "search_api"
    ...     / "search_request.v1.json"
    ... )
    >>> assert_model_roundtrip(SearchRequest, example_path)
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    k: int = 10
    filters: dict[str, JsonValue] | None = None
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

    Examples
    --------
    >>> from pathlib import Path
    >>> from kgfoundry_common.schema_helpers import assert_model_roundtrip
    >>> example_path = (
    ...     Path(__file__).parent.parent.parent
    ...     / "schema"
    ...     / "examples"
    ...     / "search_api"
    ...     / "search_result.v1.json"
    ... )
    >>> assert_model_roundtrip(SearchResult, example_path)
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    chunk_id: str
    title: str
    section: str
    score: float
    signals: dict[str, float] = Field(default_factory=dict)
    spans: dict[str, int] = Field(default_factory=dict)
    concepts: list[dict[str, str]] = Field(default_factory=list)
