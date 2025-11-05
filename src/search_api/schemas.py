"""Overview of schemas.

This module bundles schemas logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
"""
# [nav:section public-api]

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from pydantic import ConfigDict, Field

from kgfoundry_common.navmap_loader import load_nav_metadata
from kgfoundry_common.pydantic import BaseModel

if TYPE_CHECKING:
    from kgfoundry_common.types import JsonValue
else:
    JsonValue = importlib.import_module("kgfoundry_common.types").JsonValue

__all__ = [
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
]
__navmap__ = load_nav_metadata(__name__, tuple(__all__))


# [nav:anchor SearchResponse]
class SearchResponse(BaseModel):
    """Search API response containing results.

    Response envelope for search API endpoints. Contains a list of
    search results with optional metadata.

    Parameters
    ----------
    results : list[SearchResult]
        List of search results, ordered by relevance score descending.
        Defaults to empty list.

    Attributes
    ----------
    model_config : ClassVar[ConfigDict]
        Pydantic model configuration dictionary (class variable).
    results : list[SearchResult]
        List of search results, ordered by relevance score descending.
    """

    model_config = ConfigDict(extra="forbid")

    results: list[SearchResult] = Field(default_factory=list)


# [nav:anchor SearchRequest]
class SearchRequest(BaseModel):
    """Search API request model.

    Pydantic model for search API requests. Validates query parameters
    and request structure according to the search API schema.

    Parameters
    ----------
    query : str
        Search query text. Must be non-empty (min_length=1).
    k : int, optional
        Number of results to return. Defaults to 10.
    filters : dict[str, object] | None, optional
        Optional filters to apply to search results. Defaults to None.
    explain : bool, optional
        Whether to include explanation metadata in results. Defaults to False.

    Attributes
    ----------
    model_config : ClassVar[ConfigDict]
        Pydantic model configuration dictionary (class variable).
    query : str
        Search query text. Must be non-empty (min_length=1).
    k : int
        Number of results to return.
    filters : dict[str, JsonValue] | None
        Optional filters to apply to search results.
    explain : bool
        Whether to include explanation metadata in results.

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
    """Search result model.

    Pydantic model representing a single search result. Contains document
    and chunk identifiers, metadata, relevance score, and optional signals,
    spans, and concept links.

    Parameters
    ----------
    doc_id : str
        Document identifier.
    chunk_id : str
        Chunk identifier within the document.
    title : str
        Document title.
    section : str
        Section name or identifier within the document.
    score : float
        Relevance score for this result.
    signals : dict[str, float], optional
        Dictionary of signal scores (e.g., "dense", "sparse", "kg").
        Defaults to empty dictionary.
    spans : dict[str, int], optional
        Dictionary of character span information. Defaults to empty dictionary.
    concepts : list[dict[str, str]], optional
        List of linked concept dictionaries. Defaults to empty list.

    Attributes
    ----------
    model_config : ClassVar[ConfigDict]
        Pydantic model configuration dictionary (class variable).
    doc_id : str
        Document identifier.
    chunk_id : str
        Chunk identifier within the document.
    title : str
        Document title.
    section : str
        Section name or identifier within the document.
    score : float
        Relevance score for this result.
    signals : dict[str, float]
        Dictionary of signal scores (e.g., "dense", "sparse", "kg").
    spans : dict[str, int]
        Dictionary of character span information.
    concepts : list[dict[str, str]]
        List of linked concept dictionaries.

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
