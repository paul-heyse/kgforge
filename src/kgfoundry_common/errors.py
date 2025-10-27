"""
Provide utilities for module.

Notes
-----
This module exposes the primary interfaces for the package.

See Also
--------
kgfoundry_common.errors
"""


from __future__ import annotations

from typing import Final

from kgfoundry_common.navmap_types import NavMap

__all__ = [
    "ChunkingError",
    "DoclingError",
    "DownloadError",
    "EmbeddingError",
    "IndexBuildError",
    "LinkerCalibrationError",
    "Neo4jError",
    "OCRTimeoutError",
    "OntologyParseError",
    "SpladeOOMError",
    "UnsupportedMIMEError",
]

__navmap__: Final[NavMap] = {
    "title": "kgfoundry_common.errors",
    "synopsis": "Common error types shared across kgfoundry",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": [
                "DownloadError",
                "UnsupportedMIMEError",
                "DoclingError",
                "OCRTimeoutError",
                "ChunkingError",
                "EmbeddingError",
                "SpladeOOMError",
                "IndexBuildError",
                "OntologyParseError",
                "LinkerCalibrationError",
                "Neo4jError",
            ],
        },
    ],
}


# [nav:anchor DownloadError]
class DownloadError(Exception):
    """
    Represent DownloadError.
    
    Attributes
    ----------
    None
        No public attributes documented.
    
    Examples
    --------
    >>> from kgfoundry_common.errors import DownloadError
    >>> result = DownloadError()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    kgfoundry_common.errors
    
    Notes
    -----
    Document class invariants and lifecycle details here.
    """
    
    
    
    

    ...


# [nav:anchor UnsupportedMIMEError]
class UnsupportedMIMEError(Exception):
    """
    Represent UnsupportedMIMEError.
    
    Attributes
    ----------
    None
        No public attributes documented.
    
    Examples
    --------
    >>> from kgfoundry_common.errors import UnsupportedMIMEError
    >>> result = UnsupportedMIMEError()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    kgfoundry_common.errors
    
    Notes
    -----
    Document class invariants and lifecycle details here.
    """
    
    
    
    

    ...


# [nav:anchor DoclingError]
class DoclingError(Exception):
    """
    Represent DoclingError.
    
    Attributes
    ----------
    None
        No public attributes documented.
    
    Examples
    --------
    >>> from kgfoundry_common.errors import DoclingError
    >>> result = DoclingError()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    kgfoundry_common.errors
    
    Notes
    -----
    Document class invariants and lifecycle details here.
    """
    
    
    
    

    ...


# [nav:anchor OCRTimeoutError]
class OCRTimeoutError(Exception):
    """
    Represent OCRTimeoutError.
    
    Attributes
    ----------
    None
        No public attributes documented.
    
    Examples
    --------
    >>> from kgfoundry_common.errors import OCRTimeoutError
    >>> result = OCRTimeoutError()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    kgfoundry_common.errors
    
    Notes
    -----
    Document class invariants and lifecycle details here.
    """
    
    
    
    

    ...


# [nav:anchor ChunkingError]
class ChunkingError(Exception):
    """
    Represent ChunkingError.
    
    Attributes
    ----------
    None
        No public attributes documented.
    
    Examples
    --------
    >>> from kgfoundry_common.errors import ChunkingError
    >>> result = ChunkingError()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    kgfoundry_common.errors
    
    Notes
    -----
    Document class invariants and lifecycle details here.
    """
    
    
    
    

    ...


# [nav:anchor EmbeddingError]
class EmbeddingError(Exception):
    """
    Represent EmbeddingError.
    
    Attributes
    ----------
    None
        No public attributes documented.
    
    Examples
    --------
    >>> from kgfoundry_common.errors import EmbeddingError
    >>> result = EmbeddingError()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    kgfoundry_common.errors
    
    Notes
    -----
    Document class invariants and lifecycle details here.
    """
    
    
    
    

    ...


# [nav:anchor SpladeOOMError]
class SpladeOOMError(Exception):
    """
    Represent SpladeOOMError.
    
    Attributes
    ----------
    None
        No public attributes documented.
    
    Examples
    --------
    >>> from kgfoundry_common.errors import SpladeOOMError
    >>> result = SpladeOOMError()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    kgfoundry_common.errors
    
    Notes
    -----
    Document class invariants and lifecycle details here.
    """
    
    
    
    

    ...


# [nav:anchor IndexBuildError]
class IndexBuildError(Exception):
    """
    Represent IndexBuildError.
    
    Attributes
    ----------
    None
        No public attributes documented.
    
    Examples
    --------
    >>> from kgfoundry_common.errors import IndexBuildError
    >>> result = IndexBuildError()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    kgfoundry_common.errors
    
    Notes
    -----
    Document class invariants and lifecycle details here.
    """
    
    
    
    

    ...


# [nav:anchor OntologyParseError]
class OntologyParseError(Exception):
    """
    Represent OntologyParseError.
    
    Attributes
    ----------
    None
        No public attributes documented.
    
    Examples
    --------
    >>> from kgfoundry_common.errors import OntologyParseError
    >>> result = OntologyParseError()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    kgfoundry_common.errors
    
    Notes
    -----
    Document class invariants and lifecycle details here.
    """
    
    
    
    

    ...


# [nav:anchor LinkerCalibrationError]
class LinkerCalibrationError(Exception):
    """
    Represent LinkerCalibrationError.
    
    Attributes
    ----------
    None
        No public attributes documented.
    
    Examples
    --------
    >>> from kgfoundry_common.errors import LinkerCalibrationError
    >>> result = LinkerCalibrationError()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    kgfoundry_common.errors
    
    Notes
    -----
    Document class invariants and lifecycle details here.
    """
    
    
    
    

    ...


# [nav:anchor Neo4jError]
class Neo4jError(Exception):
    """
    Represent Neo4jError.
    
    Attributes
    ----------
    None
        No public attributes documented.
    
    Examples
    --------
    >>> from kgfoundry_common.errors import Neo4jError
    >>> result = Neo4jError()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    kgfoundry_common.errors
    
    Notes
    -----
    Document class invariants and lifecycle details here.
    """
    
    
    
    

    ...
