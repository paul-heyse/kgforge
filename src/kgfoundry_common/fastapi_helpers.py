"""Typed FastAPI helper utilities with structured logging and timeouts.

The helpers in this module wrap FastAPI primitives so that dependency
injection, middleware, and exception handlers retain precise type
information while also emitting kgfoundry-standard structured logs and
respecting correlation identifiers. All operations enforce a configurable
timeout to prevent runaway tasks inside the request lifecycle.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar, cast

from fastapi import Depends, FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response
from starlette.types import ASGIApp

from kgfoundry_common.logging import get_correlation_id, get_logger, with_fields

__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "typed_dependency",
    "typed_exception_handler",
    "typed_middleware",
]

DEFAULT_TIMEOUT_SECONDS = 10.0
"""Default timeout applied to FastAPI helpers (in seconds)."""

logger = get_logger(__name__)

T = TypeVar("T")
E = TypeVar("E", bound=Exception)


async def _await_with_timeout[T](coro: Awaitable[T], timeout: float | None) -> T:
    if timeout is None:
        return await coro
    return await asyncio.wait_for(coro, timeout)


async def _run_callable(
    fn: Callable[..., Awaitable[T] | T],
    timeout: float | None,
    /,
    *args: object,
    **kwargs: object,
) -> T:
    coroutine_function = cast(Callable[..., object], fn)
    if inspect.iscoroutinefunction(coroutine_function):
        async_fn = cast(Callable[..., Awaitable[T]], fn)
        return await _await_with_timeout(async_fn(*args, **kwargs), timeout)

    sync_fn = cast(Callable[..., T], fn)
    return await _await_with_timeout(asyncio.to_thread(sync_fn, *args, **kwargs), timeout)


def typed_dependency(
    dependency: Callable[..., Awaitable[T] | T],
    *,
    name: str,
    timeout: float | None = DEFAULT_TIMEOUT_SECONDS,
) -> object:
    """Return a dependency marker suitable for ``Annotated`` parameters.

    The wrapped dependency records structured logs, includes any correlation ID
    stored in :mod:`kgfoundry_common.logging`, and enforces ``timeout``.
    """

    async def _instrumented(*args: object, **kwargs: object) -> T:
        correlation_id = get_correlation_id()
        with with_fields(logger, operation=name, correlation_id=correlation_id) as log:
            start = time.perf_counter()
            log.info("dependency.start", extra={"status": "started"})
            try:
                result = await _run_callable(dependency, timeout, *args, **kwargs)
            except Exception:  # pragma: no cover - propagated to caller
                duration_ms = (time.perf_counter() - start) * 1000.0
                log.exception(
                    "dependency.error",
                    extra={"status": "error", "duration_ms": duration_ms},
                )
                raise
            duration_ms = (time.perf_counter() - start) * 1000.0
            log.info(
                "dependency.success",
                extra={"status": "success", "duration_ms": duration_ms},
            )
            return result

    dependency_callable = cast(Callable[..., Awaitable[T]], _instrumented)
    return Depends(dependency_callable)


def typed_exception_handler(
    app: FastAPI,
    exception_type: type[E],
    handler: Callable[[Request, E], Awaitable[Response] | Response],
    *,
    name: str,
    timeout: float | None = DEFAULT_TIMEOUT_SECONDS,
) -> None:
    """Register ``handler`` for ``exception_type`` with logging and timeouts."""

    async def _wrapped(request: Request, exc: E) -> Response:
        correlation_id = get_correlation_id()
        with with_fields(logger, operation=name, correlation_id=correlation_id) as log:
            start = time.perf_counter()
            exception_name = cast(
                str,
                getattr(exception_type, "__name__", exception_type.__class__.__name__),
            )
            log.info(
                "exception_handler.start",
                extra={"status": "started", "exception_type": exception_name},
            )
            try:
                result = await _run_callable(handler, timeout, request, exc)
            except Exception:  # pragma: no cover - FastAPI surfaces this
                duration_ms = (time.perf_counter() - start) * 1000.0
                log.exception(
                    "exception_handler.error",
                    extra={"status": "error", "duration_ms": duration_ms},
                )
                raise
            duration_ms = (time.perf_counter() - start) * 1000.0
            log.info(
                "exception_handler.success",
                extra={"status": "success", "duration_ms": duration_ms},
            )
            return result

    handler_callable = cast(Callable[[Request, Exception], Awaitable[Response]], _wrapped)
    app.add_exception_handler(exception_type, handler_callable)


def typed_middleware(
    app: FastAPI,
    middleware_class: type[BaseHTTPMiddleware],
    *,
    name: str,
    timeout: float | None = DEFAULT_TIMEOUT_SECONDS,
    **options: object,
) -> None:
    """Register ``middleware_class`` with instrumentation and timeouts."""
    options_copy = dict(options)

    class _InstrumentedMiddleware(BaseHTTPMiddleware):
        def __init__(self, app: ASGIApp) -> None:
            self._delegate = middleware_class(app, **options_copy)
            super().__init__(app)

        async def dispatch(  # type: ignore[override]
            self,
            request: StarletteRequest,
            call_next: Callable[[StarletteRequest], Awaitable[Response]],
        ) -> Response:
            correlation_id = get_correlation_id()
            with with_fields(logger, operation=name, correlation_id=correlation_id) as log:
                start = time.perf_counter()
                log.info("middleware.start", extra={"status": "started"})
                try:
                    response = await _await_with_timeout(
                        self._delegate.dispatch(request, call_next),
                        timeout,
                    )
                except Exception:  # pragma: no cover - propagated to caller
                    duration_ms = (time.perf_counter() - start) * 1000.0
                    log.exception(
                        "middleware.error",
                        extra={"status": "error", "duration_ms": duration_ms},
                    )
                    raise

                duration_ms = (time.perf_counter() - start) * 1000.0
                log.info(
                    "middleware.success",
                    extra={"status": "success", "duration_ms": duration_ms},
                )
                return response

    _InstrumentedMiddleware.__name__ = middleware_class.__name__
    app.add_middleware(_InstrumentedMiddleware)
