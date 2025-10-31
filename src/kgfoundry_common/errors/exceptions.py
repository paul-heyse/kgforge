"""Typed exception hierarchy with Problem Details support.

All kgfoundry exceptions inherit from KgFoundryError, which provides
structured fields and RFC 9457 Problem Details mapping.

Examples
--------
>>> from kgfoundry_common.errors import DownloadError, ErrorCode
>>> try:
...     raise DownloadError("Failed to fetch resource", cause=IOError("Connection refused"))
... except DownloadError as e:
...     assert e.code == ErrorCode.DOWNLOAD_FAILED
...     assert e.http_status == 503
...     details = e.to_problem_details(instance="/api/download")
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import cast

from kgfoundry_common.errors.codes import ErrorCode, get_type_uri
from kgfoundry_common.logging import get_logger
from kgfoundry_common.problem_details import JsonValue, ProblemDetails, build_problem_details

logger = get_logger(__name__)

__all__ = [
    "ChunkingError",
    "ConfigurationError",
    "DeserializationError",
    "DoclingError",
    "DownloadError",
    "EmbeddingError",
    "IndexBuildError",
    "KgFoundryError",
    "LinkerCalibrationError",
    "Neo4jError",
    "OCRTimeoutError",
    "OntologyParseError",
    "RetryExhaustedError",
    "SchemaValidationError",
    "SerializationError",
    "SettingsError",
    "SpladeOOMError",
    "UnsupportedMIMEError",
    "VectorSearchError",
]


class KgFoundryError(Exception):
    """Base exception for all kgfoundry errors.

    Provides structured fields (code, http_status, log_level) and
    RFC 9457 Problem Details mapping.

    Parameters
    ----------
    message : str
        Human-readable error message.
    code : ErrorCode, optional
        Error code from the registry. Defaults to RUNTIME_ERROR.
    http_status : int, optional
        HTTP status code for API responses. Defaults to 500.
    log_level : int, optional
        Logging level (logging.ERROR, logging.WARNING, etc.).
        Defaults to logging.ERROR.
    cause : Exception | None, optional
        Original exception that caused this error. Preserved via __cause__.
    context : dict[str, Any] | None, optional
        Additional context for error reporting.

    Examples
    --------
    >>> from kgfoundry_common.errors import KgFoundryError, ErrorCode
    >>> error = KgFoundryError("Operation failed", ErrorCode.RUNTIME_ERROR)
    >>> assert error.code == ErrorCode.RUNTIME_ERROR
    >>> details = error.to_problem_details(instance="/api/operation")
    >>> assert details["status"] == 500
    """

    def __init__(  # noqa: PLR0913
        self,
        message: str,
        code: ErrorCode = ErrorCode.RUNTIME_ERROR,
        http_status: int = 500,
        log_level: int = logging.ERROR,
        cause: Exception | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize error with structured fields."""
        super().__init__(message)
        self.message = message
        self.code = code
        self.http_status = http_status
        self.log_level = log_level
        self.context = dict(context) if context else {}
        if cause is not None:
            self.__cause__ = cause

    def to_problem_details(
        self,
        instance: str | None = None,
        title: str | None = None,
    ) -> ProblemDetails:
        """Convert to RFC 9457 Problem Details JSON.

        Parameters
        ----------
        instance : str | None, optional
            URI identifying the specific occurrence. Defaults to None.
        title : str | None, optional
            Short summary. Defaults to the exception class name.

        Returns
        -------
        ProblemDetails
            Problem Details object with type, title, status, detail, code,
            instance, and optional errors fields.

        Examples
        --------
        >>> error = KgFoundryError("Not found", ErrorCode.RESOURCE_UNAVAILABLE, http_status=404)
        >>> details = error.to_problem_details(instance="/api/resource/123")
        >>> assert details["type"] == "https://kgfoundry.dev/problems/resource-unavailable"
        >>> assert details["status"] == 404
        >>> assert details["code"] == "resource-unavailable"
        """
        return build_problem_details(
            type=get_type_uri(self.code),
            title=title or self.__class__.__name__,
            status=self.http_status,
            detail=self.message,
            instance=instance or "urn:kgfoundry:error",
            code=self.code.value,
            extensions=cast(Mapping[str, JsonValue] | None, self.context if self.context else None),
        )

    def __str__(self) -> str:
        """Return formatted error string."""
        base = f"{self.__class__.__name__}[{self.code.value}]: {self.message}"
        if self.__cause__:
            base += f" (caused by: {type(self.__cause__).__name__})"
        return base


class DownloadError(KgFoundryError):
    """Error during download or resource fetch operations.

    Examples
    --------
    >>> raise DownloadError("Failed to download PDF", cause=IOError("Connection refused"))
    """

    def __init__(
        self,
        message: str,
        cause: Exception | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize download error."""
        super().__init__(
            message,
            code=ErrorCode.DOWNLOAD_FAILED,
            http_status=503,
            cause=cause,
            context=context,
        )


class UnsupportedMIMEError(KgFoundryError):
    """Error for unsupported MIME types.

    Examples
    --------
    >>> raise UnsupportedMIMEError("application/x-unknown is not supported")
    """

    def __init__(
        self,
        message: str,
        cause: Exception | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize unsupported MIME error."""
        super().__init__(
            message,
            code=ErrorCode.UNSUPPORTED_MIME,
            http_status=415,
            cause=cause,
            context=context,
        )


class DoclingError(KgFoundryError):
    """Error during document processing with Docling.

    Examples
    --------
    >>> raise DoclingError("Failed to parse document", cause=ValueError("Invalid format"))
    """

    def __init__(
        self,
        message: str,
        cause: Exception | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize docling error."""
        super().__init__(
            message,
            code=ErrorCode.DOCLING_ERROR,
            http_status=422,
            cause=cause,
            context=context,
        )


class OCRTimeoutError(KgFoundryError):
    """Error when OCR operation times out.

    Examples
    --------
    >>> raise OCRTimeoutError("OCR timed out after 30s")
    """

    def __init__(
        self,
        message: str,
        cause: Exception | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize OCR timeout error."""
        super().__init__(
            message,
            code=ErrorCode.OCR_TIMEOUT,
            http_status=504,
            cause=cause,
            context=context,
        )


class ChunkingError(KgFoundryError):
    """Error during text chunking operations.

    Examples
    --------
    >>> raise ChunkingError("Failed to chunk document", cause=ValueError("Empty text"))
    """

    def __init__(
        self,
        message: str,
        cause: Exception | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize chunking error."""
        super().__init__(
            message,
            code=ErrorCode.CHUNKING_ERROR,
            http_status=422,
            cause=cause,
            context=context,
        )


class EmbeddingError(KgFoundryError):
    """Error during embedding generation.

    Examples
    --------
    >>> raise EmbeddingError("Failed to generate embeddings", cause=RuntimeError("GPU unavailable"))
    """

    def __init__(
        self,
        message: str,
        cause: Exception | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize embedding error."""
        super().__init__(
            message,
            code=ErrorCode.EMBEDDING_ERROR,
            http_status=503,
            cause=cause,
            context=context,
        )


class SpladeOOMError(KgFoundryError):
    """Error when SPLADE operation runs out of memory.

    Examples
    --------
    >>> raise SpladeOOMError("SPLADE OOM during inference")
    """

    def __init__(
        self,
        message: str,
        cause: Exception | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize SPLADE OOM error."""
        super().__init__(
            message,
            code=ErrorCode.SPLADE_OOM,
            http_status=507,
            cause=cause,
            context=context,
        )


class IndexBuildError(KgFoundryError):
    """Error during index construction.

    Examples
    --------
    >>> raise IndexBuildError("Failed to build FAISS index", cause=IOError("Disk full"))
    """

    def __init__(
        self,
        message: str,
        cause: Exception | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize index build error."""
        super().__init__(
            message,
            code=ErrorCode.INDEX_BUILD_ERROR,
            http_status=500,
            cause=cause,
            context=context,
        )


class OntologyParseError(KgFoundryError):
    """Error during ontology parsing.

    Examples
    --------
    >>> raise OntologyParseError("Failed to parse OWL file", cause=XMLSyntaxError("Invalid XML"))
    """

    def __init__(
        self,
        message: str,
        cause: Exception | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize ontology parse error."""
        super().__init__(
            message,
            code=ErrorCode.ONTOLOGY_PARSE_ERROR,
            http_status=422,
            cause=cause,
            context=context,
        )


class LinkerCalibrationError(KgFoundryError):
    """Error during linker calibration.

    Examples
    --------
    >>> raise LinkerCalibrationError("Calibration failed", cause=ValueError("Invalid parameters"))
    """

    def __init__(
        self,
        message: str,
        cause: Exception | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize linker calibration error."""
        super().__init__(
            message,
            code=ErrorCode.LINKER_CALIBRATION_ERROR,
            http_status=500,
            cause=cause,
            context=context,
        )


class Neo4jError(KgFoundryError):
    """Error during Neo4j operations.

    Examples
    --------
    >>> raise Neo4jError("Neo4j query failed", cause=ConnectionError("Database unreachable"))
    """

    def __init__(
        self,
        message: str,
        cause: Exception | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize Neo4j error."""
        super().__init__(
            message,
            code=ErrorCode.NEO4J_ERROR,
            http_status=503,
            cause=cause,
            context=context,
        )


class ConfigurationError(KgFoundryError):
    """Error during configuration validation or loading.

    Examples
    --------
    >>> raise ConfigurationError("Missing required env var: KGFOUNDRY_API_KEY")
    """

    def __init__(
        self,
        message: str,
        cause: Exception | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize configuration error."""
        super().__init__(
            message,
            code=ErrorCode.CONFIGURATION_ERROR,
            http_status=500,
            log_level=logging.CRITICAL,
            cause=cause,
            context=context,
        )


class SerializationError(KgFoundryError):
    """Error during JSON serialization or schema validation.

    Examples
    --------
    >>> raise SerializationError("Schema validation failed", cause=ValueError("Invalid type"))
    """

    def __init__(
        self,
        message: str,
        cause: Exception | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize serialization error."""
        super().__init__(
            message,
            code=ErrorCode.SERIALIZATION_ERROR,
            http_status=500,
            cause=cause,
            context=context,
        )


class SchemaValidationError(KgFoundryError):
    """Error during JSON Schema validation.

    Examples
    --------
    >>> raise SchemaValidationError("Schema validation failed", cause=ValueError("Invalid type"))
    """

    def __init__(
        self,
        message: str,
        cause: Exception | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize schema validation error."""
        super().__init__(
            message,
            code=ErrorCode.SCHEMA_VALIDATION_ERROR,
            http_status=400,
            cause=cause,
            context=context,
        )


class SettingsError(KgFoundryError):
    """Error during settings loading or validation.

    Examples
    --------
    >>> raise SettingsError("Missing required environment variable KGFOUNDRY_API_KEY")
    """

    def __init__(
        self,
        message: str,
        cause: Exception | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize settings error."""
        super().__init__(
            message,
            code=ErrorCode.CONFIGURATION_ERROR,
            http_status=500,
            cause=cause,
            context=context,
        )


class DeserializationError(KgFoundryError):
    """Error during JSON deserialization, schema validation, or checksum verification.

    Examples
    --------
    >>> raise DeserializationError("Checksum mismatch", cause=ValueError("Corrupted data"))
    """

    def __init__(
        self,
        message: str,
        cause: Exception | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize deserialization error."""
        super().__init__(
            message,
            code=ErrorCode.DESERIALIZATION_ERROR,
            http_status=500,
            cause=cause,
            context=context,
        )


class RetryExhaustedError(KgFoundryError):
    """Raised when retry logic exhausts all attempts.

    This exception indicates that a retryable operation has exhausted
    all retry attempts and should surface Problem Details with retry
    guidance information.
    """

    def __init__(
        self,
        message: str,
        *,
        operation: str | None = None,
        attempts: int | None = None,
        last_error: Exception | None = None,
        retry_after_seconds: int | None = None,
    ) -> None:
        """Initialize retry exhausted error.

        Parameters
        ----------
        message : str
            Error message describing the failure.
        operation : str | None, optional
            Name of the operation that failed.
        attempts : int | None, optional
            Number of retry attempts that were made.
        last_error : Exception | None, optional
            The last exception that occurred before retries were exhausted.
        retry_after_seconds : int | None, optional
            Suggested retry delay in seconds.
        """
        super().__init__(
            message,
            code=ErrorCode.RETRY_EXHAUSTED,
            http_status=503,
            log_level=logging.ERROR,
        )
        self.operation = operation
        self.attempts = attempts
        self.last_error = last_error
        self.retry_after_seconds = retry_after_seconds

    def to_problem_details(
        self,
        instance: str | None = None,
        title: str | None = None,
    ) -> ProblemDetails:
        """Convert to RFC 9457 Problem Details JSON.

        Parameters
        ----------
        instance : str | None, optional
            Instance URI for the specific error occurrence.
        title : str | None, optional
            Short summary. Defaults to the exception class name.

        Returns
        -------
        ProblemDetails
            Problem Details JSON structure.
        """
        extensions: dict[str, object] = {}
        if self.operation:
            extensions["operation"] = self.operation
        if self.attempts is not None:
            extensions["attempts"] = self.attempts
        if self.retry_after_seconds is not None:
            extensions["retry_after_seconds"] = self.retry_after_seconds

        return build_problem_details(
            type=get_type_uri(self.code),
            title=title or self.__class__.__name__,
            status=self.http_status,
            detail=self.message,
            instance=instance or "urn:kgfoundry:error",
            code=self.code.value,
            extensions=cast(Mapping[str, JsonValue] | None, extensions if extensions else None),
        )


class VectorSearchError(KgFoundryError):
    """Error during vector search operations.

    Examples
    --------
    >>> raise VectorSearchError("Search failed", cause=RuntimeError("Index not loaded"))
    """

    def __init__(
        self,
        message: str,
        cause: Exception | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize vector search error."""
        super().__init__(
            message,
            code=ErrorCode.VECTOR_SEARCH_ERROR,
            http_status=503,
            cause=cause,
            context=context,
        )
