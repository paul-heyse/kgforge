"""Error hierarchy for documentation build failures.

This module defines typed exceptions for documentation pipeline failures,
following RFC 9457 Problem Details patterns.
"""

from __future__ import annotations

__all__ = [
    "CatalogBuildError",
    "DocumentationBuildError",
    "GraphBuildError",
    "PortalRenderError",
    "SchemaBuildError",
    "TestMapBuildError",
]


class DocumentationBuildError(RuntimeError):
    """Base exception for all documentation build failures."""


class CatalogBuildError(DocumentationBuildError):
    """Raised when agent catalog generation fails."""


class GraphBuildError(DocumentationBuildError):
    """Raised when graph generation fails."""


class TestMapBuildError(DocumentationBuildError):
    """Raised when test map generation fails."""


class SchemaBuildError(DocumentationBuildError):
    """Raised when schema export fails."""


class PortalRenderError(DocumentationBuildError):
    """Raised when agent portal rendering fails."""
