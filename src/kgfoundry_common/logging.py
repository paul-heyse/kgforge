"""Structured logging helpers with correlation IDs and observability support.

This module provides LoggerAdapter for structured logging with mandatory
fields (correlation_id, operation, status, duration_ms) and module-level
loggers with NullHandler to prevent duplicate handlers in libraries.

Examples
--------
>>> from kgfoundry_common.logging import get_logger
>>> logger = get_logger(__name__)
>>> logger.info("Operation started", extra={"operation": "search", "status": "started"})
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Final

from kgfoundry_common.navmap_types import NavMap

__all__ = [
    "CorrelationContext",
    "JsonFormatter",
    "LoggerAdapter",
    "get_correlation_id",
    "get_logger",
    "set_correlation_id",
    "setup_logging",
    "with_fields",
]

__navmap__: Final[NavMap] = {
    "title": "kgfoundry_common.logging",
    "synopsis": "Structured logging helpers with correlation IDs and observability",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@kgfoundry-common",
        "stability": "stable",
        "since": "0.1.0",
    },
    "symbols": {
        name: {
            "owner": "@kgfoundry-common",
            "stability": "stable",
            "since": "0.1.0",
        }
        for name in __all__
    },
}

# Context variable for correlation ID propagation (async-safe)
_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None
)


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging.

    Formats log records as JSON with timestamp, level, name, message,
    and structured fields (correlation_id, operation, status, duration_ms).

    Examples
    --------
    >>> import logging
    >>> handler = logging.StreamHandler()
    >>> handler.setFormatter(JsonFormatter())
    >>> logger = logging.getLogger("test")
    >>> logger.addHandler(handler)
    >>> logger.info("Test message", extra={"operation": "test", "status": "success"})
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.

        Parameters
        ----------
        record : logging.LogRecord
            Log record to format.

        Returns
        -------
        str
            JSON-encoded log entry.
        """
        data: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }

        # Extract structured fields from extra (fields set via LoggerAdapter or extra dict)
        structured_fields = ["correlation_id", "operation", "status", "duration_ms"]
        for field in structured_fields:
            value = getattr(record, field, None)
            if value is not None:
                data[field] = value

        # Add correlation_id from context if not in extra
        if "correlation_id" not in data:
            ctx_correlation_id = _correlation_id.get()
            if ctx_correlation_id is not None:
                data["correlation_id"] = ctx_correlation_id

        # Add any additional extra fields (from extra dict passed to log calls)
        # Standard logging attributes to exclude
        excluded = {
            "name",
            "msg",
            "args",
            "created",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "message",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "thread",
            "threadName",
            "exc_info",
            "exc_text",
            "stack_info",
            "getMessage",
            "ts",  # Not a standard attribute
        }
        # Include all extra fields from record attributes (excluding standard logging attributes)
        for key, value in record.__dict__.items():
            if (
                key not in excluded
                and key not in data
                and not key.startswith("_")
                and value is not None
                and isinstance(value, (str, int, float, bool, list, dict))
            ):
                data[key] = value

        return json.dumps(data, default=str)


class LoggerAdapter(logging.LoggerAdapter):  # type: ignore[type-arg]  # pyrefly doesn't support GenericAlias
    """Logger adapter that injects structured context fields.

    This adapter ensures that all log entries include correlation_id,
    operation, status, and duration_ms fields. It propagates correlation
    IDs from context variables for async-safe operation.

    Examples
    --------
    >>> from kgfoundry_common.logging import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Search started", extra={"operation": "search", "status": "started"})
    >>> # Correlation ID is automatically injected from context
    """

    def process(
        self, msg: str, kwargs: Any
    ) -> tuple[str, Any]:  # LoggerAdapter uses Any
        """Process log message and inject structured fields.

        Parameters
        ----------
        msg : str
            Log message.
        kwargs : Any
            Keyword arguments (includes 'extra' dict from logging).

        Returns
        -------
        tuple[str, Any]
            Processed message and kwargs with injected fields.
        """
        if not isinstance(kwargs, dict):
            return msg, kwargs

        extra = kwargs.setdefault("extra", {})

        # Merge self.extra (fields from LoggerAdapter constructor) into extra
        # This allows with_fields to inject fields that persist across log calls
        if isinstance(self.extra, dict):
            for key, value in self.extra.items():
                if key not in extra:
                    extra[key] = value

        # Inject correlation_id from context if not provided
        if "correlation_id" not in extra:
            ctx_correlation_id = _correlation_id.get()
            if ctx_correlation_id is not None:
                extra["correlation_id"] = ctx_correlation_id

        # Ensure operation and status are present (defaults if missing)
        self._ensure_operation_and_status(extra, kwargs)

        return msg, kwargs

    def _ensure_operation_and_status(self, extra: dict[str, Any], kwargs: dict[str, Any]) -> None:
        """Ensure operation and status fields are present in extra dict.

        Parameters
        ----------
        extra : dict[str, Any]
            Extra dict to populate.
        kwargs : dict[str, Any]
            Keyword arguments containing log level.
        """
        if "operation" not in extra:
            extra["operation"] = "unknown"
        if "status" not in extra:
            # Infer status from log level
            level = kwargs.get("level", logging.INFO)
            if level >= logging.ERROR:
                extra["status"] = "error"
            elif level >= logging.WARNING:
                extra["status"] = "warning"
            else:
                extra["status"] = "success"


def get_logger(name: str) -> LoggerAdapter:
    """Get a logger adapter with structured logging support.

    Module-level loggers use NullHandler to prevent duplicate handlers
    in libraries. Applications should configure handlers via setup_logging().

    Parameters
    ----------
    name : str
        Logger name (typically __name__).

    Returns
    -------
    LoggerAdapter
        Logger adapter with structured context injection.

    Examples
    --------
    >>> from kgfoundry_common.logging import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Operation complete", extra={"operation": "index_build", "status": "success"})
    """
    logger = logging.getLogger(name)

    # Add NullHandler if no handlers exist (prevents duplicate handlers in libraries)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())

    return LoggerAdapter(logger, {})


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with JSON formatter.

    This function sets up structured JSON logging to stdout. It should
    be called once at application startup.

    Parameters
    ----------
    level : int, optional
        Logging level (default: logging.INFO).

    Examples
    --------
    >>> from kgfoundry_common.logging import setup_logging
    >>> import logging
    >>> setup_logging(level=logging.DEBUG)
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=level, handlers=[handler], force=True)


def set_correlation_id(correlation_id: str | None) -> None:
    """Set correlation ID in context for async propagation.

    This function uses `contextvars.ContextVar` to ensure correlation IDs
    propagate correctly through async tasks and thread pools without
    cross-contamination between concurrent requests.

    Parameters
    ----------
    correlation_id : str | None
        Correlation ID to set (or None to clear).

    Examples
    --------
    >>> from kgfoundry_common.logging import set_correlation_id, get_logger
    >>> set_correlation_id("req-123")
    >>> logger = get_logger(__name__)
    >>> logger.info("Request started")  # correlation_id="req-123" auto-injected

    Notes
    -----
    - **Async propagation**: Correlation IDs automatically propagate through
      async tasks via `contextvars.ContextVar`, ensuring each concurrent
      request maintains its own correlation ID.
    - **Thread safety**: ContextVar is thread-safe and isolates correlation
      IDs between different threads/async tasks.
    - **Cancellation**: If an async task is cancelled, the correlation ID
      context is automatically cleaned up.
    """
    _correlation_id.set(correlation_id)


def get_correlation_id() -> str | None:
    """Get current correlation ID from context.

    Returns
    -------
    str | None
        Current correlation ID or None if not set.

    Examples
    --------
    >>> from kgfoundry_common.logging import set_correlation_id, get_correlation_id
    >>> set_correlation_id("req-123")
    >>> assert get_correlation_id() == "req-123"
    """
    return _correlation_id.get()


class CorrelationContext:
    """Context manager for correlation ID propagation using contextvars.

    This class manages correlation ID context using `contextvars.ContextVar`,
    ensuring IDs propagate correctly through async tasks and thread pools
    without cross-contamination between concurrent requests.

    Examples
    --------
    >>> from kgfoundry_common.logging import CorrelationContext, get_logger
    >>> logger = get_logger(__name__)
    >>> with CorrelationContext(correlation_id="req-123"):
    ...     logger.info("Request started")  # correlation_id="req-123" auto-injected
    >>> # Correlation ID is automatically cleared when context exits
    """

    def __init__(self, correlation_id: str | None) -> None:
        """Initialize correlation context.

        Parameters
        ----------
        correlation_id : str | None
            Correlation ID to set in context (or None to clear).
        """
        self.correlation_id = correlation_id
        self._token: contextvars.Token[str | None] | None = None

    def __enter__(self) -> CorrelationContext:
        """Enter correlation context and set correlation ID.

        Returns
        -------
        CorrelationContext
            Self for use as context manager.
        """
        self._token = _correlation_id.set(self.correlation_id)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:  # noqa: ANN401
        """Exit correlation context and restore previous correlation ID.

        Parameters
        ----------
        exc_type : Any
            Exception type (if any).
        exc_val : Any
            Exception value (if any).
        exc_tb : Any
            Exception traceback (if any).
        """
        if self._token is not None:
            _correlation_id.reset(self._token)


@contextmanager
def with_fields(
    logger: logging.Logger | LoggerAdapter,
    **fields: object,
) -> Iterator[LoggerAdapter]:
    """Context manager for attaching structured fields to log entries.

    This function provides a context manager that:
    1. Sets correlation_id in contextvars if provided
    2. Returns a LoggerAdapter with bound fields
    3. Automatically restores correlation_id when context exits

    Parameters
    ----------
    logger : logging.Logger | LoggerAdapter
        Base logger to wrap (may already be an adapter).
    **fields : object
        Structured fields to inject into all log entries (e.g., correlation_id, operation, status).

    Yields
    ------
    LoggerAdapter
        Logger adapter with bound fields and correlation_id in context.

    Examples
    --------
    >>> from kgfoundry_common.logging import get_logger, with_fields
    >>> logger = get_logger(__name__)
    >>> with with_fields(logger, correlation_id="req-123", operation="build") as adapter:
    ...     adapter.info("Processing files", extra={"file_count": 10})
    ...     # correlation_id="req-123" and operation="build" are auto-injected
    >>> # Correlation ID is automatically cleared when context exits

    Notes
    -----
    - **Correlation ID propagation**: If `correlation_id` is provided in fields,
      it is set in contextvars for async propagation, then restored when the
      context exits.
    - **Field merging**: Fields provided to `with_fields` are merged with
      fields in `extra` dicts passed to log calls.
    - **NullHandler**: Libraries should use `get_logger()` which adds NullHandler
      to prevent duplicate handlers in applications.
    """
    # Extract underlying logger if already wrapped
    base_logger = logger.logger if isinstance(logger, LoggerAdapter) else logger

    # Set correlation_id in context if provided
    correlation_id = fields.get("correlation_id")
    correlation_token: contextvars.Token[str | None] | None = None
    if correlation_id is not None and isinstance(correlation_id, str):
        correlation_token = _correlation_id.set(correlation_id)

    try:
        # Create adapter with fields in extra dict
        adapter = LoggerAdapter(base_logger, fields)
        yield adapter
    finally:
        # Restore previous correlation_id when context exits
        if correlation_token is not None:
            _correlation_id.reset(correlation_token)
