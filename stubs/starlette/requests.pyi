"""Type stubs for Starlette Request."""

from __future__ import annotations

from collections.abc import Mapping

from starlette.datastructures import URL
from starlette.types import Receive, Scope, Send

__all__ = ["HTTPConnection", "Request"]

class HTTPConnection:
    """HTTP connection abstraction used by Request objects."""

    def __init__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Initialize HTTP connection."""
        ...

class Request:
    """Starlette Request object with precise type annotations."""

    headers: Mapping[str, str]
    url: URL
    scope: Scope
    _base_url: URL | None
    _client: HTTPConnection | None

    def __init__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Initialize request."""
        ...

    def get(self, key: str, default: str | None = None) -> str | None:
        """Get header value."""
        ...
