"""Overview of models.

This module bundles models logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
"""

from __future__ import annotations

from typing import Final, Literal

from pydantic import Field

from kgfoundry_common.navmap_types import NavMap
from kgfoundry_common.pydantic import BaseModel

__all__ = ["Chunk", "Doc", "DoctagsAsset", "Id", "LinkAssertion"]

__navmap__: Final[NavMap] = {
    "title": "kgfoundry_common.models",
    "synopsis": "Typed models shared across kgfoundry services",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": ["Id", "Doc", "DoctagsAsset", "Chunk", "LinkAssertion"],
        },
    ],
    "module_meta": {
        "owner": "@kgfoundry-common",
        "stability": "stable",
        "since": "0.1.0",
    },
    "symbols": {
        "Id": {
            "owner": "@kgfoundry-common",
            "stability": "stable",
            "since": "0.1.0",
        },
        "Doc": {
            "owner": "@kgfoundry-common",
            "stability": "stable",
            "since": "0.1.0",
        },
        "DoctagsAsset": {
            "owner": "@kgfoundry-common",
            "stability": "stable",
            "since": "0.1.0",
        },
        "Chunk": {
            "owner": "@kgfoundry-common",
            "stability": "stable",
            "since": "0.1.0",
        },
        "LinkAssertion": {
            "owner": "@kgfoundry-common",
            "stability": "stable",
            "since": "0.1.0",
        },
    },
}

# [nav:anchor Id]
type Id = str


# [nav:anchor Doc]
class Doc(BaseModel):
    """Describe Doc.

    <!-- auto:docstring-builder v1 -->

    the behaviour it provides to callers. Callers interact with
    validated data through this model.

    Parameters
    ----------
    id : Id
        Describe ``id``.
    openalex_id : str | None, optional
        Describe ``openalex_id``.
        Defaults to ``None``.
    doi : str | None, optional
        Describe ``doi``.
        Defaults to ``None``.
    arxiv_id : str | None, optional
        Describe ``arxiv_id``.
        Defaults to ``None``.
    pmcid : str | None, optional
        Describe ``pmcid``.
        Defaults to ``None``.
    title : str, optional
        Describe ``title``.
        Defaults to ``''``.
    authors : list[str], optional
        Describe ``authors``.
        Defaults to ``<factory>``.
    pub_date : str | None, optional
        Describe ``pub_date``.
        Defaults to ``None``.
    license : str | None, optional
        Describe ``license``.
        Defaults to ``None``.
    language : str | None, optional
        Describe ``language``.
        Defaults to ``'en'``.
    pdf_uri : str, optional
        Describe ``pdf_uri``.
        Defaults to ``''``.
    source : str, optional
        Describe ``source``.
        Defaults to ``'unknown'``.
    content_hash : str | None, optional
        Describe ``content_hash``.
        Defaults to ``None``.
    """

    id: Id
    openalex_id: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    pmcid: str | None = None
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    pub_date: str | None = None
    license: str | None = None
    language: str | None = "en"
    pdf_uri: str = ""
    source: str = "unknown"
    content_hash: str | None = None


# [nav:anchor DoctagsAsset]
class DoctagsAsset(BaseModel):
    """Describe DoctagsAsset.

    <!-- auto:docstring-builder v1 -->

    the behaviour it provides to callers. Callers interact with
    validated data through this model.

    Parameters
    ----------
    doc_id : Id
        Describe ``doc_id``.
    doctags_uri : str
        Describe ``doctags_uri``.
    pages : int
        Describe ``pages``.
    vlm_model : str
        Describe ``vlm_model``.
    vlm_revision : str
        Describe ``vlm_revision``.
    avg_logprob : float | None, optional
        Describe ``avg_logprob``.
        Defaults to ``None``.
    """

    doc_id: Id
    doctags_uri: str
    pages: int
    vlm_model: str
    vlm_revision: str
    avg_logprob: float | None = None


# [nav:anchor Chunk]
class Chunk(BaseModel):
    """Describe Chunk.

    <!-- auto:docstring-builder v1 -->

    the behaviour it provides to callers. Callers interact with
    validated data through this model.

    Parameters
    ----------
    id : Id
        Describe ``id``.
    doc_id : Id
        Describe ``doc_id``.
    section : str | None
        Describe ``section``.
    start_char : int
        Describe ``start_char``.
    end_char : int
        Describe ``end_char``.
    tokens : int
        Describe ``tokens``.
    doctags_span : dict[str, int]
        Describe ``doctags_span``.
    """

    id: Id
    doc_id: Id
    section: str | None
    start_char: int
    end_char: int
    tokens: int
    doctags_span: dict[str, int]


# [nav:anchor LinkAssertion]
class LinkAssertion(BaseModel):
    """Describe LinkAssertion.

    <!-- auto:docstring-builder v1 -->

    the behaviour it provides to callers. Callers interact with
    validated data through this model.

    Parameters
    ----------
    id : Id
        Describe ``id``.
    chunk_id : Id
        Describe ``chunk_id``.
    concept_id : Id
        Describe ``concept_id``.
    score : float
        Describe ``score``.
    decision : Literal['link', 'reject', 'uncertain']
        Describe ``decision``.
    evidence_span : str | None, optional
        Describe ``evidence_span``.
        Defaults to ``None``.
    features : dict[str, float], optional
        Describe ``features``.
        Defaults to ``<factory>``.
    run_id : str
        Describe ``run_id``.
    """

    id: Id
    chunk_id: Id
    concept_id: Id
    score: float
    decision: Literal["link", "reject", "uncertain"]
    evidence_span: str | None = None
    features: dict[str, float] = Field(default_factory=dict)
    run_id: str
