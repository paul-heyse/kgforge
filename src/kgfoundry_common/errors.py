"""Provide utilities for module.

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
    """Represent DownloadError."""
    
    
    
    
    
    
    
    
    
    
    
    
    

    ...


# [nav:anchor UnsupportedMIMEError]
class UnsupportedMIMEError(Exception):
    """Represent UnsupportedMIMEError."""
    
    
    
    
    
    
    
    
    
    
    
    
    

    ...


# [nav:anchor DoclingError]
class DoclingError(Exception):
    """Represent DoclingError."""
    
    
    
    
    
    
    
    
    
    
    
    
    

    ...


# [nav:anchor OCRTimeoutError]
class OCRTimeoutError(Exception):
    """Represent OCRTimeoutError."""
    
    
    
    
    
    
    
    
    
    
    
    
    

    ...


# [nav:anchor ChunkingError]
class ChunkingError(Exception):
    """Represent ChunkingError."""
    
    
    
    
    
    
    
    
    
    
    
    
    

    ...


# [nav:anchor EmbeddingError]
class EmbeddingError(Exception):
    """Represent EmbeddingError."""
    
    
    
    
    
    
    
    
    
    
    
    
    

    ...


# [nav:anchor SpladeOOMError]
class SpladeOOMError(Exception):
    """Represent SpladeOOMError."""
    
    
    
    
    
    
    
    
    
    
    
    
    

    ...


# [nav:anchor IndexBuildError]
class IndexBuildError(Exception):
    """Represent IndexBuildError."""
    
    
    
    
    
    
    
    
    
    
    
    
    

    ...


# [nav:anchor OntologyParseError]
class OntologyParseError(Exception):
    """Represent OntologyParseError."""
    
    
    
    
    
    
    
    
    
    
    
    
    

    ...


# [nav:anchor LinkerCalibrationError]
class LinkerCalibrationError(Exception):
    """Represent LinkerCalibrationError."""
    
    
    
    
    
    
    
    
    
    
    
    
    

    ...


# [nav:anchor Neo4jError]
class Neo4jError(Exception):
    """Represent Neo4jError."""
    
    
    
    
    
    
    
    
    
    
    
    
    

    ...
