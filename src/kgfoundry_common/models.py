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
    """Document metadata captured by the registry services.
<!-- auto:docstring-builder v1 -->

    Attributes
    ----------
    id : Id
        Internal identifier for the document record.
    openalex_id : str | None
        Optional OpenAlex identifier associated with the document.
    doi : str | None
        Digital Object Identifier string, if available.
    arxiv_id : str | None
        Optional arXiv identifier for preprint content.
    pmcid : str | None
        Optional PubMed Central identifier.
    title : str
        Human-readable title for the document.
    authors : list[str]
        Ordered list of author names.
    pub_date : str | None
        Publication date formatted as an ISO string when supplied.
    license : str | None
        License or usage rights flag associated with the document.
    language : str | None
        ISO language code representing the document language. Defaults to ``"en"``.
    pdf_uri : str
        URI pointing to the primary PDF asset.
    source : str
        Source system or ingestion pipeline that produced the record.
    content_hash : str | None
        Optional checksum for deduplication and change detection.
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
    """Metadata describing generated doctags artefacts.
<!-- auto:docstring-builder v1 -->

    Attributes
    ----------
    doc_id : Id
        Document identifier tied to the doctags asset.
    doctags_uri : str
        URI where the doctags artefact is stored.
    pages : int
        Number of pages covered by the doctags generation process.
    vlm_model : str
        Name of the vision-language model that produced the tags.
    vlm_revision : str
        Revision of the VLM model used during inference.
    avg_logprob : float | None
        Optional average log probability reported by the VLM.
    """

    doc_id: Id
    doctags_uri: str
    pages: int
    vlm_model: str
    vlm_revision: str
    avg_logprob: float | None = None


# [nav:anchor Chunk]
class Chunk(BaseModel):
    """Chunk metadata describing a passage extracted from a document.
<!-- auto:docstring-builder v1 -->

    Attributes
    ----------
    id : Id
        Identifier for the chunk record.
    doc_id : Id
        Identifier of the document that produced the chunk.
    section : str | None
        Optional section label associated with the chunk.
    start_char : int
        Inclusive start character offset within the source document.
    end_char : int
        Exclusive end character offset within the source document.
    tokens : int
        Token count used for downstream budgeting.
    doctags_span : dict[str, int]
        Mapping describing the span within doctags artefacts.
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
    """Assertion linking a chunk to a knowledge-graph concept.
<!-- auto:docstring-builder v1 -->

    Attributes
    ----------
    id : Id
        Identifier for the assertion event.
    chunk_id : Id
        Chunk that the assertion refers to.
    concept_id : Id
        Target concept identifier referenced by the link.
    score : float
        Confidence score produced by the linker.
    decision : Literal['link', 'reject', 'uncertain']
        Decision outcome assigned to the assertion.
    evidence_span : str | None
        Optional textual span used as supporting evidence.
    features : dict[str, float]
        Per-feature scores contributing to the decision.
    run_id : str
        Pipeline run identifier that produced the assertion.
    """

    id: Id
    chunk_id: Id
    concept_id: Id
    score: float
    decision: Literal["link", "reject", "uncertain"]
    evidence_span: str | None = None
    features: dict[str, float] = Field(default_factory=dict)
    run_id: str
