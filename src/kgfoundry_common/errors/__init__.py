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

from typing import Final

from kgfoundry_common.errors.codes import BASE_TYPE_URI, ErrorCode, get_type_uri
from kgfoundry_common.errors.exceptions import (
    AgentCatalogSearchError,
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
    UnsupportedMIMEError,
    VectorSearchError,
)
from kgfoundry_common.navmap_types import NavMap

# HTTP adapters are optional (require fastapi)
_http_exports: list[str] = []
try:
    from kgfoundry_common.errors.http import (  # noqa: F401
        problem_details_response,
        register_problem_details_handler,
    )

    _http_exports = ["problem_details_response", "register_problem_details_handler"]
except ImportError:
    pass  # FastAPI not available

__all__ = [
    "AgentCatalogSearchError",
    "BASE_TYPE_URI",
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
    "RetryExhaustedError",
    "RegistryError",
    "SchemaValidationError",
    "SerializationError",
    "SettingsError",
    "SpladeOOMError",
    "UnsupportedMIMEError",
    "VectorSearchError",
    "get_type_uri",
    *_http_exports,
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
