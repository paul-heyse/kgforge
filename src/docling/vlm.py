"""
Provide utilities for module.

Notes
-----
This module exposes the primary interfaces for the package.

See Also
--------
docling.vlm
"""


from __future__ import annotations

from typing import Final

from kgfoundry_common.navmap_types import NavMap

__all__ = ["GraniteDoclingVLM"]

__navmap__: Final[NavMap] = {
    "title": "docling.vlm",
    "synopsis": "Vision-language tagging helpers for docling",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": ["GraniteDoclingVLM"],
        },
    ],
}


# [nav:anchor GraniteDoclingVLM]
class GraniteDoclingVLM:
    """
    Represent GraniteDoclingVLM.
    
    Attributes
    ----------
    None
        No public attributes documented.
    
    Examples
    --------
    >>> from docling.vlm import GraniteDoclingVLM
    >>> result = GraniteDoclingVLM()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    docling.vlm
    
    Notes
    -----
    Document class invariants and lifecycle details here.
    """
    
    
    
    

    ...
