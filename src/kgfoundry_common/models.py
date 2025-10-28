"""Provide utilities for module.

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
    """Describe Doc."""
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

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
    """Describe DoctagsAsset."""
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    doc_id: Id
    doctags_uri: str
    pages: int
    vlm_model: str
    vlm_revision: str
    avg_logprob: float | None = None


# [nav:anchor Chunk]
class Chunk(BaseModel):
    """Describe Chunk."""
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    id: Id
    doc_id: Id
    section: str | None
    start_char: int
    end_char: int
    tokens: int
    doctags_span: dict[str, int]


# [nav:anchor LinkAssertion]
class LinkAssertion(BaseModel):
    """Describe LinkAssertion."""
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    id: Id
    chunk_id: Id
    concept_id: Id
    score: float
    decision: Literal["link", "reject", "uncertain"]
    evidence_span: str | None = None
    features: dict[str, float] = {}
    run_id: str
