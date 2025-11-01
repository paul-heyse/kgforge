"""Type stubs for Starlette middleware module."""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware

__all__ = ["BaseHTTPMiddleware", "Middleware"]

class Middleware:
    """Middleware decorator/class for Starlette applications.

    This is a type alias or wrapper used for middleware registration.
    """

    def __init__(
        self,
        middleware_class: type[BaseHTTPMiddleware],
        *args: object,
        **kwargs: object,
    ) -> None:
        """Initialize middleware wrapper.

        Parameters
        ----------
        middleware_class : type[BaseHTTPMiddleware]
            Middleware class to instantiate.
        *args : Any
            Positional arguments for middleware initialization.
        **kwargs : Any
            Keyword arguments for middleware initialization.
        """
        ...
