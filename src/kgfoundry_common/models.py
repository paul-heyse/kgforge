"""Overview of models.

This module bundles models logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
"""


from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel

from kgfoundry_common.navmap_types import NavMap

__all__ = ["Chunk", "Doc", "DoctagsAsset", "LinkAssertion"]

__navmap__: Final[NavMap] = {
    "title": "kgfoundry_common.models",
    "synopsis": "Typed models shared across kgfoundry services",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": ["Doc", "DoctagsAsset", "Chunk", "LinkAssertion"],
        },
    ],
    "module_meta": {
        "owner": "@kgfoundry-common",
        "stability": "stable",
        "since": "0.1.0",
    },
    "symbols": {
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

Id = str


# [nav:anchor Doc]
class Doc(BaseModel):
    """Model the Doc.

    Pydantic model defining the structured payload used across the system. Validation ensures inputs
    conform to the declared schema while producing clear error messages. Use this class when
    serialising or parsing data for the surrounding feature.
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
    """Model the DoctagsAsset.

    Pydantic model defining the structured payload used across the system. Validation ensures inputs
    conform to the declared schema while producing clear error messages. Use this class when
    serialising or parsing data for the surrounding feature.
    """
    

    doc_id: Id
    doctags_uri: str
    pages: int
    vlm_model: str
    vlm_revision: str
    avg_logprob: float | None = None


# [nav:anchor Chunk]
class Chunk(BaseModel):
    """Model the Chunk.

    Pydantic model defining the structured payload used across the system. Validation ensures inputs
    conform to the declared schema while producing clear error messages. Use this class when
    serialising or parsing data for the surrounding feature.
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
    """Model the LinkAssertion.

    Pydantic model defining the structured payload used across the system. Validation ensures inputs
    conform to the declared schema while producing clear error messages. Use this class when
    serialising or parsing data for the surrounding feature.
    """
    

    id: Id
    chunk_id: Id
    concept_id: Id
    score: float
    decision: Literal["link", "reject", "uncertain"]
    evidence_span: str | None = None
    features: dict[str, float] = {}
    run_id: str
