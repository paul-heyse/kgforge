"""
Provide utilities for module.

Notes
-----
This module exposes the primary interfaces for the package.

See Also
--------
docling.hybrid
"""


from __future__ import annotations

from typing import Final

from kgfoundry_common.navmap_types import NavMap

__all__ = ["HybridChunker"]

__navmap__: Final[NavMap] = {
    "title": "docling.hybrid",
    "synopsis": "Hybrid docling pipeline combining layout and text cues",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": ["HybridChunker"],
        },
    ],
}


# [nav:anchor HybridChunker]
class HybridChunker:
    """
    Represent HybridChunker.
    
    Attributes
    ----------
    None
        No public attributes documented.
    
    Examples
    --------
    >>> from docling.hybrid import HybridChunker
    >>> result = HybridChunker()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    docling.hybrid
    
    Notes
    -----
    Document class invariants and lifecycle details here.
    """
    
    
    
    

    ...
