
from __future__ import annotations
from pydantic import BaseModel
from typing import Optional, List, Dict, Literal

Id = str

class Doc(BaseModel):
    id: Id
    openalex_id: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    pmcid: Optional[str] = None
    title: str = ""
    authors: List[str] = []
    pub_date: Optional[str] = None
    license: Optional[str] = None
    language: Optional[str] = "en"
    pdf_uri: str = ""
    source: str = "unknown"
    content_hash: Optional[str] = None

class DoctagsAsset(BaseModel):
    doc_id: Id
    doctags_uri: str
    pages: int
    vlm_model: str
    vlm_revision: str
    avg_logprob: Optional[float] = None

class Chunk(BaseModel):
    id: Id
    doc_id: Id
    section: Optional[str]
    start_char: int
    end_char: int
    tokens: int
    doctags_span: Dict[str, int]

class LinkAssertion(BaseModel):
    id: Id
    chunk_id: Id
    concept_id: Id
    score: float
    decision: Literal['link','reject','uncertain']
    evidence_span: Optional[str] = None
    features: Dict[str, float] = {}
    run_id: str
