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
# [nav:section public-api]

from __future__ import annotations

from enum import StrEnum
from typing import Final

from kgfoundry_common.navmap_loader import load_nav_metadata

__all__ = [
    "BASE_TYPE_URI",
    "ErrorCode",
    "get_type_uri",
]
__navmap__ = load_nav_metadata(__name__, tuple(__all__))


# [nav:anchor BASE_TYPE_URI]
BASE_TYPE_URI: Final[str] = "https://kgfoundry.dev/problems"


# [nav:anchor ErrorCode]
class ErrorCode(StrEnum):
    """Stable error codes for kgfoundry exceptions.

    Enumeration of error codes used in RFC 9457 Problem Details responses.
    Codes follow kebab-case naming and remain stable across releases to ensure
    backward compatibility.

    Error codes are organized by category:
    - 1xx: Download & Ingestion
    - 2xx: Document Processing
    - 3xx: Embedding & Indexing
    - 4xx: Search & Retrieval
    - 5xx: Configuration & Runtime
    - 6xx: Knowledge Graph & Ontology
    - 7xx: Serialization & Persistence

    Attributes
    ----------
    DOWNLOAD_FAILED : ErrorCode
        Download operation failed.
    UNSUPPORTED_MIME : ErrorCode
        Unsupported MIME type.
    INVALID_INPUT : ErrorCode
        Invalid input provided.
    DOCLING_ERROR : ErrorCode
        Document parsing error.
    OCR_TIMEOUT : ErrorCode
        OCR operation timed out.
    CHUNKING_ERROR : ErrorCode
        Document chunking error.
    EMBEDDING_ERROR : ErrorCode
        Embedding generation error.
    INDEX_BUILD_ERROR : ErrorCode
        Index build error.
    SPLADE_OOM : ErrorCode
        SPLADE out of memory error.
    RETRY_EXHAUSTED : ErrorCode
        Retry attempts exhausted.
    SEARCH_INDEX_MISSING : ErrorCode
        Search index not found.
    SEARCH_QUERY_INVALID : ErrorCode
        Invalid search query.
    SEARCH_TIMEOUT : ErrorCode
        Search operation timed out.
    VECTOR_SEARCH_ERROR : ErrorCode
        Vector search error.
    AGENT_CATALOG_SEARCH_ERROR : ErrorCode
        Agent catalog search error.
    CATALOG_LOAD_ERROR : ErrorCode
        Catalog load error.
    SYMBOL_ATTACHMENT_ERROR : ErrorCode
        Symbol attachment error.
    CONFIGURATION_ERROR : ErrorCode
        Configuration error.
    RUNTIME_ERROR : ErrorCode
        Runtime error.
    RESOURCE_UNAVAILABLE : ErrorCode
        Resource unavailable.
    SESSION_ERROR : ErrorCode
        Session error.
    ONTOLOGY_PARSE_ERROR : ErrorCode
        Ontology parsing error.
    LINKER_CALIBRATION_ERROR : ErrorCode
        Linker calibration error.
    NEO4J_ERROR : ErrorCode
        Neo4j database error.
    SERIALIZATION_ERROR : ErrorCode
        Serialization error.
    DESERIALIZATION_ERROR : ErrorCode
        Deserialization error.
    SCHEMA_VALIDATION_ERROR : ErrorCode
        Schema validation error.
    REGISTRY_ERROR : ErrorCode
        Registry error.
    ARTIFACT_MODEL_ERROR : ErrorCode
        Artifact model error.
    ARTIFACT_VALIDATION_ERROR : ErrorCode
        Artifact validation error.
    ARTIFACT_SERIALIZATION_ERROR : ErrorCode
        Artifact serialization error.
    ARTIFACT_DESERIALIZATION_ERROR : ErrorCode
        Artifact deserialization error.
    ARTIFACT_DEPENDENCY_ERROR : ErrorCode
        Artifact dependency error.

    Examples
    --------
    >>> code = ErrorCode.DOWNLOAD_FAILED
    >>> assert code == "download-failed"
    >>> assert isinstance(code, ErrorCode)
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
    CATALOG_LOAD_ERROR = "catalog-load-error"
    SYMBOL_ATTACHMENT_ERROR = "symbol-attachment-error"

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
    REGISTRY_ERROR = "registry-error"
    ARTIFACT_MODEL_ERROR = "artifact-model-error"
    ARTIFACT_VALIDATION_ERROR = "artifact-validation-error"
    ARTIFACT_SERIALIZATION_ERROR = "artifact-serialization-error"
    ARTIFACT_DESERIALIZATION_ERROR = "artifact-deserialization-error"
    ARTIFACT_DEPENDENCY_ERROR = "artifact-dependency-error"

    def __str__(self) -> str:
        """Return the code value as a string.

        Returns
        -------
        str
            The error code value (e.g., "download-failed").
        """
        return self.value


# [nav:anchor get_type_uri]
def get_type_uri(code: ErrorCode) -> str:
    """Get the RFC 9457 type URI for an error code.

    Constructs a complete type URI by combining BASE_TYPE_URI with the
    error code value. Type URIs are used in Problem Details responses
    to uniquely identify error types.

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
