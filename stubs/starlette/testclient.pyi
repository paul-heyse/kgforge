"""Type stubs for Starlette testclient module."""

from __future__ import annotations

from typing import Any

__all__ = ["TestClient"]

class HTTPConnection:
    """HTTP connection abstraction used by TestClient."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize HTTP connection."""
        ...

class TestClient:
    """Starlette test client for testing ASGI applications."""

    def __init__(
        self,
        app: Any,  # ASGI application instance
        base_url: str = "http://test",
        raise_server_exceptions: bool = True,
        root_path: str = "",
        backend: str = "asyncio",
        backend_options: dict[str, Any] | None = None,
    ) -> None:
        """Initialize test client.

        Parameters
        ----------
        app : Any
            ASGI application instance.
        base_url : str, optional
            Base URL for requests. Defaults to "http://test".
        raise_server_exceptions : bool, optional
            Whether to raise exceptions from server. Defaults to True.
        root_path : str, optional
            Root path for requests. Defaults to "".
        backend : str, optional
            Backend for async operations. Defaults to "asyncio".
        backend_options : dict[str, Any] | None, optional
            Options for backend. Defaults to None.
        """
        ...

    def get(
        self,
        url: str,
        **kwargs: Any,
    ) -> Any:  # Returns Response-like object
        """Make GET request."""
        ...

    def post(
        self,
        url: str,
        **kwargs: Any,
    ) -> Any:  # Returns Response-like object
        """Make POST request."""
        ...

    def put(
        self,
        url: str,
        **kwargs: Any,
    ) -> Any:  # Returns Response-like object
        """Make PUT request."""
        ...

    def delete(
        self,
        url: str,
        **kwargs: Any,
    ) -> Any:  # Returns Response-like object
        """Make DELETE request."""
        ...

    def patch(
        self,
        url: str,
        **kwargs: Any,
    ) -> Any:  # Returns Response-like object
        """Make PATCH request."""
        ...

    def options(
        self,
        url: str,
        **kwargs: Any,
    ) -> Any:  # Returns Response-like object
        """Make OPTIONS request."""
        ...

    def head(
        self,
        url: str,
        **kwargs: Any,
    ) -> Any:  # Returns Response-like object
        """Make HEAD request."""
        ...
