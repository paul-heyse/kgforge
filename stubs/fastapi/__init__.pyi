from __future__ import annotations

from collections.abc import Callable
from typing import Any

import fastapi
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

__all__ = ["Depends", "FastAPI", "HTTPException", "Header", "Request"]

class Depends:
    """FastAPI dependency injection."""

    def __init__(self, dependency: Callable[..., Any]) -> None:
        """Initialize dependency."""
        ...

class Header:
    """FastAPI header parameter."""

    def __init__(self, default: Any = ..., **kwargs: Any) -> None:
        """Initialize header parameter."""
        ...

class HTTPException(Exception):
    """FastAPI HTTP exception."""

    def __init__(self, status_code: int, detail: str, **kwargs: Any) -> None:
        """Initialize HTTP exception."""
        ...

class FastAPI(fastapi.FastAPI):  # type: ignore[misc]  # FastAPI is a runtime class
    """FastAPI application with precise type annotations."""

    def __init__(
        self,
        *,
        title: str = "FastAPI",
        version: str = "0.0.0",
        **kwargs: Any,
    ) -> None:
        """Initialize FastAPI application with typed parameters."""
        ...

    def exception_handler(
        self, exc_class_or_status_code: type[Exception] | int
    ) -> Callable[[Callable[..., object]], Callable[..., object]]:
        """Register exception handler decorator.

        Returns a decorator that preserves the original function signature.
        Starlette's implementation uses Callable without type arguments.
        """
        ...

    def get(
        self,
        path: str,
        *,
        response_model: type[Any] | None = None,
        **kwargs: Any,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register GET route decorator.

        Returns a decorator that takes a route handler function and returns it unchanged.
        The handler signature is preserved for type checking.
        """
        ...

    def post(
        self,
        path: str,
        *,
        response_model: type[Any] | None = None,
        **kwargs: Any,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register POST route decorator.

        Returns a decorator that takes a route handler function and returns it unchanged.
        The handler signature is preserved for type checking.
        """
        ...

    def middleware(
        self,
        middleware_type: str,
    ) -> Callable[
        [type[BaseHTTPMiddleware] | Callable[[FastAPI], BaseHTTPMiddleware]],
        type[BaseHTTPMiddleware] | Callable[[FastAPI], BaseHTTPMiddleware],
    ]:
        """Register middleware decorator.

        Returns a decorator that takes a middleware class or factory function and returns it unchanged.
        """
        ...
