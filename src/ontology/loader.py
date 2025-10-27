"""
Provide utilities for module.

Notes
-----
This module exposes the primary interfaces for the package.

See Also
--------
ontology.loader
"""


from __future__ import annotations

from typing import Final

from kgfoundry_common.navmap_types import NavMap

__all__ = ["OntologyLoader"]

__navmap__: Final[NavMap] = {
    "title": "ontology.loader",
    "synopsis": "Module for ontology.loader",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": ["OntologyLoader"],
        },
    ],
}


# [nav:anchor OntologyLoader]
class OntologyLoader:
    """
    Represent OntologyLoader.
    
    Attributes
    ----------
    None
        No public attributes documented.
    
    Examples
    --------
    >>> from ontology.loader import OntologyLoader
    >>> result = OntologyLoader()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    ontology.loader
    
    Notes
    -----
    Document class invariants and lifecycle details here.
    """
    
    
    
    

    ...
