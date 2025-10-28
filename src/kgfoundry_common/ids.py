"""Provide utilities for module.

Notes
-----
This module exposes the primary interfaces for the package.

See Also
--------
kgfoundry_common.ids
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
    """Return urn doc from text.

    Parameters
    ----------
    text : str
        Description for ``text``.
    
    Returns
    -------
    str
        Description of return value.
    
    Examples
    --------
    >>> from kgfoundry_common.ids import urn_doc_from_text
    >>> result = urn_doc_from_text(...)
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    kgfoundry_common.ids
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
    """
    
    h = hashlib.sha256(text.encode("utf-8")).digest()[:16]
    b32 = base64.b32encode(h).decode("ascii").strip("=").lower()
    return f"urn:doc:sha256:{b32}"


# [nav:anchor urn_chunk]
def urn_chunk(doc_hash: str, start: int, end: int) -> str:
    """Return urn chunk.

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
    
    Examples
    --------
    >>> from kgfoundry_common.ids import urn_chunk
    >>> result = urn_chunk(..., ..., ...)
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    kgfoundry_common.ids
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
    """
    
    return f"urn:chunk:{doc_hash.split(':')[-1]}:{start}-{end}"
