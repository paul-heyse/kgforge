"""Observability instrumentation for CodeIntel.

This module provides Prometheus metrics and tracing instrumentation for CodeIntel
operations. It aligns with AGENTS.md principle 9: "Emit structured logs, Prometheus
metrics, and OpenTelemetry traces at boundaries."

All metrics follow Prometheus naming conventions:
- Counters: _total suffix
- Histograms: _seconds/_bytes suffix
- Gauges: no suffix, represent current state

Metrics are designed to answer key operational questions:
- What is the request rate per tool?
- What is the p50/p95/p99 latency per tool?
- How many requests are rate-limited?
- What is the current index size?
- What is the error rate by error type?
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from functools import wraps
from typing import Any, ParamSpec, TypeVar

from anyio.lowlevel import checkpoint
from prometheus_client import Counter, Gauge, Histogram
from tools import get_logger

_logger = get_logger(__name__)

# === MCP Tool Metrics ===

TOOL_CALLS_TOTAL = Counter(
    "codeintel_tool_calls_total",
    "Total MCP tool calls by tool name and final status",
    ["tool", "status"],  # status: success, error, timeout, rate_limited, cancelled
)

TOOL_DURATION_SECONDS = Histogram(
    "codeintel_tool_duration_seconds",
    "MCP tool execution duration in seconds",
    ["tool"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

TOOL_ERRORS_TOTAL = Counter(
    "codeintel_tool_errors_total",
    "Total MCP tool errors by tool name and error type",
    ["tool", "error_type"],
)

# === Rate Limiting Metrics ===

RATE_LIMIT_REJECTIONS_TOTAL = Counter(
    "codeintel_rate_limit_rejections_total",
    "Total requests rejected by rate limiter",
)

RATE_LIMIT_BUCKET_TOKENS = Gauge(
    "codeintel_rate_limit_bucket_tokens",
    "Current number of tokens in rate limit bucket",
)

# === Index Metrics ===

INDEX_SIZE_SYMBOLS = Gauge(
    "codeintel_index_symbols_total",
    "Total symbols in persistent index by language",
    ["lang"],
)

INDEX_SIZE_REFS = Gauge(
    "codeintel_index_refs_total",
    "Total references in persistent index",
)

INDEX_SIZE_FILES = Gauge(
    "codeintel_index_files_total",
    "Total files in persistent index",
)

INDEX_BUILD_DURATION_SECONDS = Histogram(
    "codeintel_index_build_duration_seconds",
    "Duration of index build operations in seconds",
    buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
)

# === Parse Metrics ===

PARSE_DURATION_SECONDS = Histogram(
    "codeintel_parse_duration_seconds",
    "Tree-sitter parse duration in seconds by language",
    ["lang"],
    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
)

PARSE_FILE_SIZE_BYTES = Histogram(
    "codeintel_parse_file_size_bytes",
    "Size of files parsed in bytes",
    buckets=[100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1048576],
)

# === MCP Protocol Metrics ===

MCP_REQUESTS_TOTAL = Counter(
    "codeintel_mcp_requests_total",
    "Total MCP JSON-RPC requests by method",
    ["method"],
)

MCP_REQUEST_DURATION_SECONDS = Histogram(
    "codeintel_mcp_request_duration_seconds",
    "MCP request processing duration in seconds",
    ["method"],
    buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 5.0],
)

# === Type Parameters for Decorators ===

P = ParamSpec("P")
R = TypeVar("R")


def instrument_tool(
    tool_name: str,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Instrument async MCP tool handlers with metrics.

    This decorator automatically tracks:
    - Total calls by status (success, error, timeout, etc.)
    - Execution duration histogram
    - Error counts by type

    Parameters
    ----------
    tool_name : str
        Name of the MCP tool (e.g., "ts.query", "code.getOutline").

    Returns
    -------
    Callable[[Callable[P, R]], Callable[P, R]]
        Decorator function that takes a callable and returns a wrapped callable
        with the same signature but instrumented with metrics tracking.

    Examples
    --------
    >>> @instrument_tool("ts.query")
    ... async def _tool_ts_query(payload):
    ...     # ... implementation ...
    ...     return result
    """

    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start = time.monotonic()
            status = "success"
            error_type = None
            try:
                return await func(*args, **kwargs)
            except Exception as exc:
                status = "error"
                error_type = type(exc).__name__
                TOOL_ERRORS_TOTAL.labels(tool=tool_name, error_type=error_type).inc()
                raise
            finally:
                duration = time.monotonic() - start
                TOOL_CALLS_TOTAL.labels(tool=tool_name, status=status).inc()
                TOOL_DURATION_SECONDS.labels(tool=tool_name).observe(duration)

                log_ctx = {
                    "tool": tool_name,
                    "status": status,
                    "duration_s": round(duration, 4),
                }
                if error_type:
                    log_ctx["error_type"] = error_type

                if status == "success":
                    _logger.debug("Tool execution completed", extra=log_ctx)
                else:
                    _logger.warning("Tool execution failed", extra=log_ctx)

        return wrapper  # type: ignore[return-value]

    return decorator


@asynccontextmanager
async def trace_operation(
    operation: str,
    *,
    log_start: bool = False,
    log_end: bool = True,
    **attributes: object,
) -> AsyncIterator[dict[str, Any]]:
    """Context manager for tracing operations with structured logging.

    This provides a lightweight tracing mechanism until OpenTelemetry is integrated.
    It logs operation start/end with duration and structured attributes.

    Parameters
    ----------
    operation : str
        Name of the operation being traced (e.g., "index.build", "parse.python").
    log_start : bool, optional
        Whether to log when operation starts, by default False.
    log_end : bool, optional
        Whether to log when operation ends, by default True.
    **attributes : object
        Additional structured attributes to include in logs.

    Yields
    ------
    dict[str, Any]
        Mutable context dictionary for adding attributes during execution.

    Examples
    --------
    >>> async with trace_operation("index.build", repo="myrepo") as ctx:
    ...     # ... do work ...
    ...     ctx["files_indexed"] = 42
    """
    start = time.monotonic()
    ctx: dict[str, Any] = {"operation": operation, **attributes}

    if log_start:
        _logger.info("Operation started: %s", operation, extra=ctx)

    # Yield control to the event loop to honour the async contract.
    await checkpoint()

    try:
        yield ctx
    finally:
        duration = time.monotonic() - start
        ctx["duration_s"] = round(duration, 4)

        if log_end:
            _logger.info("Operation completed: %s", operation, extra=ctx)


def instrument_sync_tool(tool_name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Instrument synchronous MCP tool handlers with metrics.

    Similar to instrument_tool but for synchronous functions.

    Parameters
    ----------
    tool_name : str
        Name of the MCP tool.

    Returns
    -------
    Callable[[Callable[P, R]], Callable[P, R]]
        Decorator function that takes a callable and returns a wrapped callable
        with the same signature but instrumented with metrics tracking.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start = time.monotonic()
            status = "success"
            error_type = None
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                status = "error"
                error_type = type(exc).__name__
                TOOL_ERRORS_TOTAL.labels(tool=tool_name, error_type=error_type).inc()
                raise
            finally:
                duration = time.monotonic() - start
                TOOL_CALLS_TOTAL.labels(tool=tool_name, status=status).inc()
                TOOL_DURATION_SECONDS.labels(tool=tool_name).observe(duration)

        return wrapper  # type: ignore[return-value]

    return decorator


def update_index_metrics(symbol_counts: dict[str, int], ref_count: int, file_count: int) -> None:
    """Update index size gauges with current statistics.

    This should be called after index build operations to keep metrics current.

    Parameters
    ----------
    symbol_counts : dict[str, int]
        Mapping from language name to symbol count.
    ref_count : int
        Total number of references in the index.
    file_count : int
        Total number of files in the index.

    Examples
    --------
    >>> update_index_metrics({"python": 1234, "json": 56}, 789, 42)
    """
    for lang, count in symbol_counts.items():
        INDEX_SIZE_SYMBOLS.labels(lang=lang).set(count)
    INDEX_SIZE_REFS.set(ref_count)
    INDEX_SIZE_FILES.set(file_count)


def record_parse(language: str, file_size: int, duration: float) -> None:
    """Record Tree-sitter parse metrics.

    Parameters
    ----------
    language : str
        Language that was parsed.
    file_size : int
        Size of the file in bytes.
    duration : float
        Parse duration in seconds.

    Examples
    --------
    >>> record_parse("python", 5000, 0.012)
    """
    PARSE_DURATION_SECONDS.labels(lang=language).observe(duration)
    PARSE_FILE_SIZE_BYTES.observe(file_size)
