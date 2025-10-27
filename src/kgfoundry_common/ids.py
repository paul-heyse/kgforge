"""Helpers for generating deterministic URNs used across kgfoundry.

NavMap:
- urn_doc_from_text: Create a document URN derived from textual content.
- urn_chunk: Create a chunk URN derived from offsets within a document.
"""

from __future__ import annotations

import base64
import hashlib
from typing import Final

from kgfoundry_common.navmap_types import NavMap

__all__ = ["urn_chunk", "urn_doc_from_text"]

__navmap__: Final[NavMap] = {
    "title": "kgfoundry_common.ids",
    "synopsis": "Helpers for generating deterministic URNs",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": ["urn_doc_from_text", "urn_chunk"],
        },
    ],
}


# [nav:anchor urn_doc_from_text]
def urn_doc_from_text(text: str) -> str:
    """Return a normalized document URN derived from ``text``.

    Parameters
    ----------
    text : str
        Raw document text to hash.

    Returns
    -------
    str
        Deterministic URN containing the truncated SHA-256 hash.
    """
    h = hashlib.sha256(text.encode("utf-8")).digest()[:16]
    b32 = base64.b32encode(h).decode("ascii").strip("=").lower()
    return f"urn:doc:sha256:{b32}"


# [nav:anchor urn_chunk]
def urn_chunk(doc_hash: str, start: int, end: int) -> str:
    """Return a chunk URN derived from document hash and offsets.

    Parameters
    ----------
    doc_hash : str
        Document-level URN or hash prefix.
    start : int
        Inclusive character offset for the chunk start.
    end : int
        Exclusive character offset for the chunk end.

    Returns
    -------
    str
        Deterministic chunk URN joining the document hash with offsets.
    """
    return f"urn:chunk:{doc_hash.split(':')[-1]}:{start}-{end}"
