"""Module for kgforge_common.errors.

NavMap:
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


class DownloadError(Exception):
    """Raised when an external download fails."""

    ...


class UnsupportedMIMEError(Exception):
    """Raised when an unsupported MIME type is encountered."""

    ...


class DoclingError(Exception):
    """Base Docling processing error."""

    ...


class OCRTimeoutError(Exception):
    """Raised when OCR operations exceed time limits."""

    ...


class ChunkingError(Exception):
    """Raised when chunk generation fails."""

    ...


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""

    ...


class SpladeOOMError(Exception):
    """Raised when SPLADE runs out of memory."""

    ...


class IndexBuildError(Exception):
    """Raised when index construction fails."""

    ...


class OntologyParseError(Exception):
    """Raised when ontology parsing fails."""

    ...


class LinkerCalibrationError(Exception):
    """Raised when linker calibration cannot be performed."""

    ...


class Neo4jError(Exception):
    """Raised when Neo4j operations fail."""

    ...
