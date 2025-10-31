"""Type stubs for Starlette Request."""

from __future__ import annotations

from typing import Any
from urllib.parse import SplitResult

__all__ = ["HTTPConnection", "Request"]

class HTTPConnection:
    """HTTP connection abstraction used by Request objects."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize HTTP connection."""
        ...

class Request:
    """Starlette Request object with precise type annotations."""

    headers: dict[str, str]
    url: SplitResult
    scope: dict[str, Any]
    _base_url: SplitResult | None
    _client: HTTPConnection | None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize request."""
        ...

    def get(self, key: str, default: str | None = None) -> str | None:
        """Get header value."""
        ...
