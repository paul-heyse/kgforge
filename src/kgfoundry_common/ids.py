"""Overview of ids.

This module bundles ids logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
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
    """Describe urn doc from text.

    <!-- auto:docstring-builder v1 -->

    Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

    Parameters
    ----------
    text : str
        Describe ``text``.

    Returns
    -------
    str
        Describe return value.
"""
    h = hashlib.sha256(text.encode("utf-8")).digest()[:16]
    b32 = base64.b32encode(h).decode("ascii").strip("=").lower()
    return f"urn:doc:sha256:{b32}"


# [nav:anchor urn_chunk]
def urn_chunk(doc_hash: str, start: int, end: int) -> str:
    """Describe urn chunk.

    <!-- auto:docstring-builder v1 -->

    Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

    Parameters
    ----------
    doc_hash : str
        Describe ``doc_hash``.
    start : int
        Describe ``start``.
    end : int
        Describe ``end``.

    Returns
    -------
    str
        Describe return value.
"""
    return f"urn:chunk:{doc_hash.split(':')[-1]}:{start}-{end}"
