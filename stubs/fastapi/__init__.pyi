from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import ParamSpec, TypeVar

import fastapi
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

__all__ = ["Depends", "FastAPI", "HTTPException", "Header", "Request", "Response", "Security"]

RouteParams = ParamSpec("RouteParams")
RouteReturn = TypeVar("RouteReturn")
DependencyReturn = TypeVar("DependencyReturn")
ExceptionReturn = TypeVar("ExceptionReturn", bound=Response)

class Depends[RouteReturn]:
    """FastAPI dependency injection marker preserving dependency signature."""

    def __init__(
        self,
        dependency: Callable[RouteParams, Awaitable[RouteReturn] | RouteReturn] | None = None,
        *,
        use_cache: bool = True,
    ) -> None:
        """Initialize dependency."""
        ...

class Security[RouteReturn](Depends[RouteReturn]):
    """FastAPI security dependency marker."""

    def __init__(
        self,
        dependency: Callable[RouteParams, Awaitable[RouteReturn] | RouteReturn],
        scopes: Sequence[str] | None = None,
        *,
        use_cache: bool = True,
    ) -> None:
        """Initialize security dependency."""
        ...

class Header:
    """FastAPI header parameter."""

    def __init__(self, default: object = ..., **kwargs: object) -> None:
        """Initialize header parameter."""
        ...

class HTTPException(Exception):  # noqa: N818 - follows FastAPI runtime name
    """FastAPI HTTP exception."""

    def __init__(self, status_code: int, detail: object, **kwargs: object) -> None:
        """Initialize HTTP exception."""
        ...

class FastAPI(fastapi.FastAPI):  # type: ignore[misc]  # FastAPI is a runtime class
    """FastAPI application with precise type annotations."""

    def __init__(
        self,
        *,
        title: str = "FastAPI",
        version: str = "0.0.0",
        **kwargs: object,
    ) -> None:
        """Initialize FastAPI application with typed parameters."""
        ...

    def exception_handler(
        self, exc_class_or_status_code: type[Exception] | int
    ) -> Callable[[Callable[RouteParams, RouteReturn]], Callable[RouteParams, RouteReturn]]:
        """Register exception handler decorator.

        Returns a decorator that preserves the original function signature.
        Starlette's implementation uses Callable without type arguments.
        """
        ...

    def add_exception_handler(
        self,
        exc_class_or_status_code: type[Exception] | int,
        handler: Callable[[Request, Exception], Awaitable[Response] | Response],
    ) -> None:
        """Register an exception handler for ``exc_class_or_status_code``."""
        ...

    def get(
        self,
        path: str,
        *,
        response_model: type[object] | None = None,
        **kwargs: object,
    ) -> Callable[
        [Callable[RouteParams, Awaitable[RouteReturn] | RouteReturn]],
        Callable[RouteParams, Awaitable[RouteReturn] | RouteReturn],
    ]:
        """Register GET route decorator.

        Returns a decorator that takes a route handler function and returns it unchanged.
        The handler signature is preserved for type checking.
        """
        ...

    def post(
        self,
        path: str,
        *,
        response_model: type[object] | None = None,
        **kwargs: object,
    ) -> Callable[
        [Callable[RouteParams, Awaitable[RouteReturn] | RouteReturn]],
        Callable[RouteParams, Awaitable[RouteReturn] | RouteReturn],
    ]:
        """Register POST route decorator.

        Returns a decorator that takes a route handler function and returns it unchanged.
        The handler signature is preserved for type checking.
        """
        ...

    def add_middleware(
        self,
        middleware_class: type[BaseHTTPMiddleware],
        **options: object,
    ) -> None:
        """Register ``middleware_class`` with optional configuration ``options``."""
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
