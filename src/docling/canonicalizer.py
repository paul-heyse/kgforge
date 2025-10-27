"""
Provide utilities for module.

Notes
-----
This module exposes the primary interfaces for the package.

See Also
--------
docling.canonicalizer
"""


from __future__ import annotations

import re
import unicodedata
from typing import Final

from kgfoundry_common.navmap_types import NavMap

__all__ = ["canonicalize_text"]

__navmap__: Final[NavMap] = {
    "title": "docling.canonicalizer",
    "synopsis": "String canonicalisation utilities for docling preprocessing",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": ["canonicalize_text"],
        },
    ],
}


# [nav:anchor canonicalize_text]
def canonicalize_text(blocks: list[str]) -> str:
    """
    Return canonicalize text.
    
    Parameters
    ----------
    blocks : List[str]
        Description for ``blocks``.
    
    Returns
    -------
    str
        Description of return value.
    
    Examples
    --------
    >>> from docling.canonicalizer import canonicalize_text
    >>> result = canonicalize_text(...)
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    docling.canonicalizer
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
    """
    
    

    def norm(s: str) -> str:
        """
        Return norm.
        
        Parameters
        ----------
        s : str
            Description for ``s``.
        
        Returns
        -------
        str
            Description of return value.
        
        Examples
        --------
        >>> from docling.canonicalizer import norm
        >>> result = norm(...)
        >>> result  # doctest: +ELLIPSIS
        ...
        
        See Also
        --------
        docling.canonicalizer
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        s = unicodedata.normalize("NFC", s)
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        s = re.sub(r"[\u2022\u25E6\u2013]", "-", s)  # bullets/dashes
        s = re.sub(r"[\x00-\x1F]", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    normed = [norm(b) for b in blocks if b.strip()]
    return "\n".join(normed)
