"""Module for kgforge_common.models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Id = str


class Doc(BaseModel):
    """Doc."""

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


class DoctagsAsset(BaseModel):
    """Doctagsasset."""

    doc_id: Id
    doctags_uri: str
    pages: int
    vlm_model: str
    vlm_revision: str
    avg_logprob: float | None = None


class Chunk(BaseModel):
    """Chunk."""

    id: Id
    doc_id: Id
    section: str | None
    start_char: int
    end_char: int
    tokens: int
    doctags_span: dict[str, int]


class LinkAssertion(BaseModel):
    """Linkassertion."""

    id: Id
    chunk_id: Id
    concept_id: Id
    score: float
    decision: Literal["link", "reject", "uncertain"]
    evidence_span: str | None = None
    features: dict[str, float] = {}
    run_id: str
