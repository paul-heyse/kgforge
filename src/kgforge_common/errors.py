"""Module for kgforge_common.errors."""


class DownloadError(Exception):
    """Raised when an external download fails."""

    ...


class UnsupportedMIMEError(Exception):
    """Raised when an unsupported MIME type is encountered."""

    ...


class DoclingError(Exception):
    """Base Docling processing error."""

    ...


class OCRTimeout(Exception):
    """Raised when OCR operations exceed time limits."""

    ...


class ChunkingError(Exception):
    """Raised when chunk generation fails."""

    ...


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""

    ...


class SpladeOOM(Exception):
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
