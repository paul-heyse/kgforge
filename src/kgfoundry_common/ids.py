"""Ids utilities."""

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
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@kgfoundry-common",
        "stability": "stable",
        "since": "0.1.0",
    },
    "symbols": {
        "urn_doc_from_text": {
            "owner": "@kgfoundry-common",
            "stability": "stable",
            "since": "0.1.0",
        },
        "urn_chunk": {
            "owner": "@kgfoundry-common",
            "stability": "stable",
            "since": "0.1.0",
        },
    },
}


# [nav:anchor urn_doc_from_text]
def urn_doc_from_text(text: str) -> str:
    """Compute urn doc from text.

    Carry out the urn doc from text operation.

    Parameters
    ----------
    text : str
        Description for ``text``.

    Returns
    -------
    str
        Description of return value.
    """
    
    
    
    
    h = hashlib.sha256(text.encode("utf-8")).digest()[:16]
    b32 = base64.b32encode(h).decode("ascii").strip("=").lower()
    return f"urn:doc:sha256:{b32}"


# [nav:anchor urn_chunk]
def urn_chunk(doc_hash: str, start: int, end: int) -> str:
    """Compute urn chunk.

    Carry out the urn chunk operation.

    Parameters
    ----------
    doc_hash : str
        Description for ``doc_hash``.
    start : int
        Description for ``start``.
    end : int
        Description for ``end``.

    Returns
    -------
    str
        Description of return value.
    """
    
    
    
    
    return f"urn:chunk:{doc_hash.split(':')[-1]}:{start}-{end}"
