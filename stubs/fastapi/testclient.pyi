"""Type stubs for FastAPI testclient module."""

from __future__ import annotations

from collections.abc import Mapping

from requests import Response
from starlette.testclient import TestClient as StarletteTestClient
from starlette.types import ASGIApp

__all__ = ["TestClient"]

class TestClient(StarletteTestClient):
    """FastAPI test client for testing ASGI applications.

    Wraps Starlette's TestClient with FastAPI-specific behavior.
    """

    def __init__(
        self,
        app: ASGIApp,
        base_url: str = "http://test",
        raise_server_exceptions: bool = True,
        root_path: str = "",
        backend: str = "asyncio",
        backend_options: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize test client.

        Parameters
        ----------
        app : Any
            FastAPI application instance or ASGI app.
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
        **kwargs: object,
    ) -> Response:
        """Make GET request."""
        ...

    def post(
        self,
        url: str,
        **kwargs: object,
    ) -> Response:
        """Make POST request."""
        ...

    def put(
        self,
        url: str,
        **kwargs: object,
    ) -> Response:
        """Make PUT request."""
        ...

    def delete(
        self,
        url: str,
        **kwargs: object,
    ) -> Response:
        """Make DELETE request."""
        ...

    def patch(
        self,
        url: str,
        **kwargs: object,
    ) -> Response:
        """Make PATCH request."""
        ...

    def options(
        self,
        url: str,
        **kwargs: object,
    ) -> Response:
        """Make OPTIONS request."""
        ...

    def head(
        self,
        url: str,
        **kwargs: object,
    ) -> Response:
        """Make HEAD request."""
        ...
