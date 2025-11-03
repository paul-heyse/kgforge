"""Exception hierarchy and Problem Details support.

This module provides typed exceptions with RFC 9457 Problem Details
mapping and structured error handling.

Examples
--------
>>> from kgfoundry_common.errors import KgFoundryError, ErrorCode
>>> try:
...     raise KgFoundryError("Operation failed", ErrorCode.RUNTIME_ERROR)
... except KgFoundryError as e:
...     details = e.to_problem_details(instance="/api/search")
...     assert details["type"] == "https://kgfoundry.dev/problems/runtime-error"
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Final, Protocol, cast

from kgfoundry_common.errors.codes import BASE_TYPE_URI, ErrorCode, get_type_uri
from kgfoundry_common.errors.exceptions import (
    AgentCatalogSearchError,
    ArtifactDependencyError,
    ArtifactDeserializationError,
    ArtifactModelError,
    ArtifactSerializationError,
    ArtifactValidationError,
    CatalogLoadError,
    CatalogSessionError,
    ChunkingError,
    ConfigurationError,
    DeserializationError,
    DoclingError,
    DownloadError,
    EmbeddingError,
    IndexBuildError,
    KgFoundryError,
    LinkerCalibrationError,
    Neo4jError,
    OCRTimeoutError,
    OntologyParseError,
    RegistryError,
    RetryExhaustedError,
    SchemaValidationError,
    SerializationError,
    SettingsError,
    SpladeOOMError,
    SymbolAttachmentError,
    UnsupportedMIMEError,
    VectorSearchError,
)
from kgfoundry_common.navmap_types import NavMap

# Structural protocols for optional FastAPI integration. We keep the
# expectations minimal so the package does not need FastAPI installed for
# static analysis while preserving typed call signatures.


class RequestProtocol(Protocol):
    """Structural type for FastAPI's Request object."""


class JSONResponseProtocol(Protocol):
    """Structural type for FastAPI's JSONResponse."""

    status_code: int


class FastAPIProtocol(Protocol):
    """Structural type for FastAPI application instances."""

    def add_exception_handler(
        self,
        exception_class: type[BaseException],
        handler: Callable[..., object],
        *,
        name: str | None = None,
    ) -> None:
        """Register an exception handler."""


class ProblemDetailsResponse(Protocol):
    """Callable protocol for Problem Details response helpers."""

    def __call__(
        self,
        error: KgFoundryError,
        request: RequestProtocol | None = None,
    ) -> JSONResponseProtocol:
        """Convert a KgFoundryError into a JSON-response payload."""
        ...


class RegisterProblemDetailsHandler(Protocol):
    """Callable protocol for registering Problem Details handlers."""

    def __call__(self, app: FastAPIProtocol) -> None:
        """Attach the Problem Details handler to a FastAPI-like application."""
        ...


# HTTP adapters are optional (require fastapi)
try:
    from kgfoundry_common.errors.http import (
        problem_details_response as _http_problem_details_response,
    )
    from kgfoundry_common.errors.http import (
        register_problem_details_handler as _http_register_problem_details_handler,
    )
except ImportError:  # pragma: no cover - optional dependency

    def _problem_details_response(
        error: KgFoundryError,
        request: RequestProtocol | None = None,
    ) -> JSONResponseProtocol:
        """Raise an informative error when FastAPI dependencies are missing."""
        message = (
            "FastAPI support is not installed. Install kgfoundry[api] to enable "
            "problem details response helpers."
        )
        raise RuntimeError(message)

    def _register_problem_details_handler(app: FastAPIProtocol) -> None:
        """Raise an informative error when FastAPI dependencies are missing."""
        message = (
            "FastAPI support is not installed. Install kgfoundry[api] to enable "
            "Problem Details handlers."
        )
        raise RuntimeError(message)
else:
    _problem_details_response = cast(
        ProblemDetailsResponse,
        _http_problem_details_response,
    )
    _register_problem_details_handler = cast(
        RegisterProblemDetailsHandler,
        _http_register_problem_details_handler,
    )


def problem_details_response(
    error: KgFoundryError,
    request: RequestProtocol | None = None,
) -> JSONResponseProtocol:
    """Convert ``KgFoundryError`` to a Problem Details response.

    This wrapper preserves the optional FastAPI dependency while exposing a
    typed interface that accepts the structural ``RequestProtocol`` defined in
    this module. When FastAPI support is installed, the helper delegates to the
    implementation in ``kgfoundry_common.errors.http``. Otherwise it raises an
    informative ``RuntimeError`` describing the missing optional dependency.
    """
    response = _problem_details_response(
        error,
        cast(Any, request),
    )
    return cast(JSONResponseProtocol, response)


def register_problem_details_handler(app: FastAPIProtocol) -> None:
    """Register the KgFoundry Problem Details handler on a FastAPI app."""
    _register_problem_details_handler(cast(Any, app))


__all__ = [
    "BASE_TYPE_URI",
    "AgentCatalogSearchError",
    "ArtifactDependencyError",
    "ArtifactDeserializationError",
    "ArtifactModelError",
    "ArtifactSerializationError",
    "ArtifactValidationError",
    "CatalogLoadError",
    "CatalogSessionError",
    "ChunkingError",
    "ConfigurationError",
    "DeserializationError",
    "DoclingError",
    "DownloadError",
    "EmbeddingError",
    "ErrorCode",
    "IndexBuildError",
    "KgFoundryError",
    "LinkerCalibrationError",
    "Neo4jError",
    "OCRTimeoutError",
    "OntologyParseError",
    "RegistryError",
    "RetryExhaustedError",
    "SchemaValidationError",
    "SerializationError",
    "SettingsError",
    "SpladeOOMError",
    "SymbolAttachmentError",
    "UnsupportedMIMEError",
    "VectorSearchError",
    "get_type_uri",
    "problem_details_response",
    "register_problem_details_handler",
]

__navmap__: Final[NavMap] = {
    "title": "kgfoundry_common.errors",
    "synopsis": "Exception hierarchy and Problem Details support",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@kgfoundry-common",
        "stability": "stable",
        "since": "0.1.0",
    },
    "symbols": {
        name: {
            "owner": "@kgfoundry-common",
            "stability": "stable",
            "since": "0.1.0",
        }
        for name in __all__
    },
}
