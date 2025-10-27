"""Module for kgfoundry_common.errors.

NavMap:
- NavMap: Structure describing a module navmap.
- DownloadError: Raised when an external download fails.
- UnsupportedMIMEError: Raised when an unsupported MIME type is encountered.
- DoclingError: Base Docling processing error.
- OCRTimeoutError: Raised when OCR operations exceed time limits.
- ChunkingError: Raised when chunk generation fails.
- EmbeddingError: Raised when embedding generation fails.
- SpladeOOMError: Raised when SPLADE runs out of memory.
- IndexBuildError: Raised when index construction fails.
- OntologyParseError: Raised when ontology parsing fails.
- LinkerCalibrationError: Raised when linker calibration cannot be performed.
- Neo4jError: Raised when Neo4j operations fail.
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
    """Raised when an external download fails."""

    ...


# [nav:anchor UnsupportedMIMEError]
class UnsupportedMIMEError(Exception):
    """Raised when an unsupported MIME type is encountered."""

    ...


# [nav:anchor DoclingError]
class DoclingError(Exception):
    """Base Docling processing error."""

    ...


# [nav:anchor OCRTimeoutError]
class OCRTimeoutError(Exception):
    """Raised when OCR operations exceed time limits."""

    ...


# [nav:anchor ChunkingError]
class ChunkingError(Exception):
    """Raised when chunk generation fails."""

    ...


# [nav:anchor EmbeddingError]
class EmbeddingError(Exception):
    """Raised when embedding generation fails."""

    ...


# [nav:anchor SpladeOOMError]
class SpladeOOMError(Exception):
    """Raised when SPLADE runs out of memory."""

    ...


# [nav:anchor IndexBuildError]
class IndexBuildError(Exception):
    """Raised when index construction fails."""

    ...


# [nav:anchor OntologyParseError]
class OntologyParseError(Exception):
    """Raised when ontology parsing fails."""

    ...


# [nav:anchor LinkerCalibrationError]
class LinkerCalibrationError(Exception):
    """Raised when linker calibration cannot be performed."""

    ...


# [nav:anchor Neo4jError]
class Neo4jError(Exception):
    """Raised when Neo4j operations fail."""

    ...
