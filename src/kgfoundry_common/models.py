"""
Provide utilities for module.

Notes
-----
This module exposes the primary interfaces for the package.

See Also
--------
kgfoundry_common.models
"""


from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel

from kgfoundry_common.navmap_types import NavMap

__all__ = ["Chunk", "Doc", "DoctagsAsset", "LinkAssertion"]

__navmap__: Final[NavMap] = {
    "title": "kgfoundry_common.models",
    "synopsis": "Module for kgfoundry_common.models",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": ["Doc", "DoctagsAsset", "Chunk", "LinkAssertion"],
        },
    ],
}

Id = str


# [nav:anchor Doc]
class Doc(BaseModel):
    """
    Represent Doc.
    
    Attributes
    ----------
    id : Id
        Attribute description.
    openalex_id : str | None
        Attribute description.
    doi : str | None
        Attribute description.
    arxiv_id : str | None
        Attribute description.
    pmcid : str | None
        Attribute description.
    title : str
        Attribute description.
    authors : List[str]
        Attribute description.
    pub_date : str | None
        Attribute description.
    license : str | None
        Attribute description.
    language : str | None
        Attribute description.
    pdf_uri : str
        Attribute description.
    source : str
        Attribute description.
    content_hash : str | None
        Attribute description.
    
    Examples
    --------
    >>> from kgfoundry_common.models import Doc
    >>> result = Doc()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    kgfoundry_common.models
    
    Notes
    -----
    Document class invariants and lifecycle details here.
    """
    
    
    
    

    id: Id
    openalex_id: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    pmcid: str | None = None
    title: str = ""
    authors: list[str] = []
    pub_date: str | None = None
    license: str | None = None
    language: str | None = "en"
    pdf_uri: str = ""
    source: str = "unknown"
    content_hash: str | None = None


# [nav:anchor DoctagsAsset]
class DoctagsAsset(BaseModel):
    """
    Represent DoctagsAsset.
    
    Attributes
    ----------
    doc_id : Id
        Attribute description.
    doctags_uri : str
        Attribute description.
    pages : int
        Attribute description.
    vlm_model : str
        Attribute description.
    vlm_revision : str
        Attribute description.
    avg_logprob : float | None
        Attribute description.
    
    Examples
    --------
    >>> from kgfoundry_common.models import DoctagsAsset
    >>> result = DoctagsAsset()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    kgfoundry_common.models
    
    Notes
    -----
    Document class invariants and lifecycle details here.
    """
    
    
    
    

    doc_id: Id
    doctags_uri: str
    pages: int
    vlm_model: str
    vlm_revision: str
    avg_logprob: float | None = None


# [nav:anchor Chunk]
class Chunk(BaseModel):
    """
    Represent Chunk.
    
    Attributes
    ----------
    id : Id
        Attribute description.
    doc_id : Id
        Attribute description.
    section : str | None
        Attribute description.
    start_char : int
        Attribute description.
    end_char : int
        Attribute description.
    tokens : int
        Attribute description.
    doctags_span : Mapping[str, int]
        Attribute description.
    
    Examples
    --------
    >>> from kgfoundry_common.models import Chunk
    >>> result = Chunk()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    kgfoundry_common.models
    
    Notes
    -----
    Document class invariants and lifecycle details here.
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
    """
    Represent LinkAssertion.
    
    Attributes
    ----------
    id : Id
        Attribute description.
    chunk_id : Id
        Attribute description.
    concept_id : Id
        Attribute description.
    score : float
        Attribute description.
    decision : Literal['link', 'reject', 'uncertain']
        Attribute description.
    evidence_span : str | None
        Attribute description.
    features : Mapping[str, float]
        Attribute description.
    run_id : str
        Attribute description.
    
    Examples
    --------
    >>> from kgfoundry_common.models import LinkAssertion
    >>> result = LinkAssertion()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    kgfoundry_common.models
    
    Notes
    -----
    Document class invariants and lifecycle details here.
    """
    
    
    
    

    id: Id
    chunk_id: Id
    concept_id: Id
    score: float
    decision: Literal["link", "reject", "uncertain"]
    evidence_span: str | None = None
    features: dict[str, float] = {}
    run_id: str
