from __future__ import annotations

from collections.abc import Callable

import fastapi
from starlette.requests import Request

__all__ = ["FastAPI", "Request"]

class FastAPI(fastapi.FastAPI):  # type: ignore[misc]  # FastAPI is a runtime class
    def exception_handler(
        self, exc_class_or_status_code: type[Exception] | int
    ) -> Callable[[Callable[..., object]], Callable[..., object]]:
        """Register exception handler decorator.

        Returns a decorator that preserves the original function signature.
        Starlette's implementation uses Callable without type arguments.
        """
