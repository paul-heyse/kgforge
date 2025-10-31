"""Legacy errors module - redirects to new error hierarchy.

This module maintains backward compatibility by re-exporting exceptions
from the new structured error hierarchy. New code should import directly
from kgfoundry_common.errors.
"""

from __future__ import annotations

from kgfoundry_common.errors.exceptions import (
    ChunkingError,
    DoclingError,
    DownloadError,
    EmbeddingError,
    IndexBuildError,
    LinkerCalibrationError,
    Neo4jError,
    OCRTimeoutError,
    OntologyParseError,
    SpladeOOMError,
    UnsupportedMIMEError,
)

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
