"""Type stubs for Starlette BaseHTTPMiddleware."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

__all__ = ["BaseHTTPMiddleware"]

class BaseHTTPMiddleware:
    """Base HTTP middleware with precise type annotations."""

    def __init__(self, app: ASGIApp) -> None:
        """Initialize middleware.

        Parameters
        ----------
        app : Any
            ASGI application instance.
        """
        ...

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Dispatch middleware request.

        Parameters
        ----------
        request : Request
            The incoming HTTP request.
        call_next : Callable[[Request], Awaitable[Response]]
            Callable that takes a Request and returns an awaitable Response.

        Returns
        -------
        Response
            The HTTP response.
        """
        ...
