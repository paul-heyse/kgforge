"""Error code registry and type URIs for Problem Details.

This module defines stable error codes and type URIs used in RFC 9457
Problem Details responses. Codes and URIs are frozen after initial release
to maintain backward compatibility.

Examples
--------
>>> from kgfoundry_common.errors.codes import ErrorCode, get_type_uri
>>> code = ErrorCode.DOWNLOAD_FAILED
>>> type_uri = get_type_uri(code)
>>> assert type_uri == "https://kgfoundry.dev/problems/download-failed"
"""

from __future__ import annotations

from enum import Enum
from typing import Final

__all__ = ["BASE_TYPE_URI", "ErrorCode", "get_type_uri"]

BASE_TYPE_URI: Final[str] = "https://kgfoundry.dev/problems"


class ErrorCode(str, Enum):
    """Stable error codes for kgfoundry exceptions.

    Codes follow kebab-case naming and remain stable across releases.
    """

    # Download & Ingestion (1xx)
    DOWNLOAD_FAILED = "download-failed"
    UNSUPPORTED_MIME = "unsupported-mime"
    INVALID_INPUT = "invalid-input"

    # Document Processing (2xx)
    DOCLING_ERROR = "docling-error"
    OCR_TIMEOUT = "ocr-timeout"
    CHUNKING_ERROR = "chunking-error"

    # Embedding & Indexing (3xx)
    EMBEDDING_ERROR = "embedding-error"
    INDEX_BUILD_ERROR = "index-build-error"
    SPLADE_OOM = "splade-oom"
    RETRY_EXHAUSTED = "retry-exhausted"

    # Search & Retrieval (4xx)
    SEARCH_INDEX_MISSING = "search-index-missing"
    SEARCH_QUERY_INVALID = "search-query-invalid"
    SEARCH_TIMEOUT = "search-timeout"
    VECTOR_SEARCH_ERROR = "vector-search-error"
    AGENT_CATALOG_SEARCH_ERROR = "agent-catalog-search-error"

    # Configuration & Runtime (5xx)
    CONFIGURATION_ERROR = "configuration-error"
    RUNTIME_ERROR = "runtime-error"
    RESOURCE_UNAVAILABLE = "resource-unavailable"
    SESSION_ERROR = "session-error"

    # Knowledge Graph & Ontology (6xx)
    ONTOLOGY_PARSE_ERROR = "ontology-parse-error"
    LINKER_CALIBRATION_ERROR = "linker-calibration-error"
    NEO4J_ERROR = "neo4j-error"

    # Serialization & Persistence (7xx)
    SERIALIZATION_ERROR = "serialization-error"
    DESERIALIZATION_ERROR = "deserialization-error"
    SCHEMA_VALIDATION_ERROR = "schema-validation-error"

    def __str__(self) -> str:
        """Return the code value."""
        return self.value


def get_type_uri(code: ErrorCode) -> str:
    """Get the RFC 9457 type URI for an error code.

    Parameters
    ----------
    code : ErrorCode
        Error code enum value.

    Returns
    -------
    str
        Complete type URI (e.g., "https://kgfoundry.dev/problems/download-failed").

    Examples
    --------
    >>> get_type_uri(ErrorCode.DOWNLOAD_FAILED)
    'https://kgfoundry.dev/problems/download-failed'
    """
    return f"{BASE_TYPE_URI}/{code.value}"
