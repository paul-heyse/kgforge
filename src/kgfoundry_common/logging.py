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
import time
from collections.abc import Mapping
from contextlib import AbstractContextManager
from types import TracebackType
from typing import TYPE_CHECKING, Any, Final, TypedDict, cast

from kgfoundry_common.navmap_types import NavMap
from kgfoundry_common.types import JsonValue

__all__ = [
    "CorrelationContext",
    "JsonFormatter",
    "LogContextExtra",
    "LoggerAdapter",
    "get_correlation_id",
    "get_logger",
    "measure_duration",
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


class LogContextExtra(TypedDict, total=False):
    """TypedDict for structured logging extra fields.

    Fields added to logging records via the 'extra' parameter to enable
    proper type checking when accessing LogRecord attributes.

    Attributes
    ----------
    correlation_id : str, optional
        Request or correlation ID for tracing across services.
    operation : str, optional
        Name of the operation being logged (e.g., "search", "index_build").
    status : str, optional
        Operation status ("success", "error", "started", "in_progress").
    duration_ms : float, optional
        Operation duration in milliseconds.
    service : str, optional
        Name of the service producing the log.
    endpoint : str, optional
        HTTP endpoint or internal method being executed.
    """

    correlation_id: str
    operation: str
    status: str
    duration_ms: float
    service: str
    endpoint: str


# Context variable for correlation ID propagation (async-safe)
_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None
)


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging.

    <!-- auto:docstring-builder v1 -->

    Formats log records as JSON with timestamp, level, name, message,
    and structured fields (correlation_id, operation, status, duration_ms).

    Parameters
    ----------
    fmt : inspect._empty, optional
        Describe ``fmt``.
        Defaults to ``None``.
    datefmt : inspect._empty, optional
        Describe ``datefmt``.
        Defaults to ``None``.
    style : inspect._empty, optional
        Describe ``style``.
        Defaults to ``'%'``.
    validate : inspect._empty, optional
        Describe ``validate``.
        Defaults to ``True``.
    defaults : inspect._empty, optional
        Describe ``defaults``.
        Defaults to ``None``.

    Examples
    --------
    >>> import logging
    >>> handler = logging.StreamHandler()
    >>> handler.setFormatter(JsonFormatter())
    >>> logger = logging.getLogger("test")
    >>> logger.addHandler(handler)
    >>> logger.info("Test message", extra={"operation": "test", "status": "success"})

    Returns
    -------
    inspect._empty
        Describe return value.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        record : logging.LogRecord
            Log record to format.

        Returns
        -------
        str
            JSON-encoded log entry.
        """
        # Log data is JSON-serializable - use JsonValue instead of Any
        data: dict[str, JsonValue] = {
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


if TYPE_CHECKING:

    class _LoggerAdapterBase(logging.LoggerAdapter):  # pragma: no cover - typing helper
        logger: logging.Logger
        extra: Mapping[str, object] | None
else:
    _LoggerAdapterBase = logging.LoggerAdapter


class LoggerAdapter(_LoggerAdapterBase):
    """Logger adapter that injects structured context fields.

    <!-- auto:docstring-builder v1 -->

    This adapter ensures that all log entries include correlation_id,
    operation, status, and duration_ms fields. It propagates correlation
    IDs from context variables for async-safe operation.

    Parameters
    ----------
    logger : logging.Logger
        Base logger instance to wrap.
    extra : Mapping[str, object] | None, optional
        Structured fields to inject into log entries.
        Defaults to ``None``.

    Examples
    --------
    >>> from kgfoundry_common.logging import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Search started", extra={"operation": "search", "status": "started"})
    >>> # Correlation ID is automatically injected from context
    """

    logger: logging.Logger
    extra: Mapping[str, object] | None

    def process(self, msg: str, kwargs: Mapping[str, Any]) -> tuple[str, Any]:
        """Process log message and inject structured fields.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        msg : str
            Log message.
        kwargs : Mapping[str, Any]
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
        self._ensure_operation_and_status(extra, kwargs.get("level", logging.INFO))

        return msg, kwargs

    @staticmethod
    def _ensure_operation_and_status(extra: dict[str, Any], level: int) -> None:
        """Ensure operation and status fields are present in extra dict.

        Parameters
        ----------
        extra : dict[str, Any]
            Extra dict to populate.
        level : int
            Log level to determine status.
        """
        if "operation" not in extra:
            extra["operation"] = "unknown"
        if "status" not in extra:
            # Infer status from log level
            if level >= logging.ERROR:
                extra["status"] = "error"
            elif level >= logging.WARNING:
                extra["status"] = "warning"
            else:
                extra["status"] = "success"

    def debug(self, msg: object, *args: object, **kwargs: object) -> None:
        """Log a debug message with structured fields."""
        kwargs_dict = cast(dict[str, Any], kwargs)
        extra = kwargs_dict.get("extra", {})
        if not isinstance(extra, dict):
            extra = {}
        # Merge self.extra (fields from LoggerAdapter constructor) into extra
        if isinstance(self.extra, dict):
            for key, value in self.extra.items():
                if key not in extra:
                    extra[key] = value
        # Inject correlation_id from context if not provided
        if "correlation_id" not in extra:
            ctx_correlation_id = _correlation_id.get()
            if ctx_correlation_id is not None:
                extra["correlation_id"] = ctx_correlation_id
        self._ensure_operation_and_status(extra, logging.DEBUG)
        kwargs_dict["extra"] = extra
        self.logger.debug(msg, *args, **kwargs_dict)

    def info(self, msg: object, *args: object, **kwargs: object) -> None:
        """Log an info message with structured fields."""
        kwargs_dict = cast(dict[str, Any], kwargs)
        extra = kwargs_dict.get("extra", {})
        if not isinstance(extra, dict):
            extra = {}
        # Merge self.extra (fields from LoggerAdapter constructor) into extra
        if isinstance(self.extra, dict):
            for key, value in self.extra.items():
                if key not in extra:
                    extra[key] = value
        # Inject correlation_id from context if not provided
        if "correlation_id" not in extra:
            ctx_correlation_id = _correlation_id.get()
            if ctx_correlation_id is not None:
                extra["correlation_id"] = ctx_correlation_id
        self._ensure_operation_and_status(extra, logging.INFO)
        kwargs_dict["extra"] = extra
        self.logger.info(msg, *args, **kwargs_dict)

    def warning(self, msg: object, *args: object, **kwargs: object) -> None:
        """Log a warning message with structured fields."""
        kwargs_dict = cast(dict[str, Any], kwargs)
        extra = kwargs_dict.get("extra", {})
        if not isinstance(extra, dict):
            extra = {}
        # Merge self.extra (fields from LoggerAdapter constructor) into extra
        if isinstance(self.extra, dict):
            for key, value in self.extra.items():
                if key not in extra:
                    extra[key] = value
        # Inject correlation_id from context if not provided
        if "correlation_id" not in extra:
            ctx_correlation_id = _correlation_id.get()
            if ctx_correlation_id is not None:
                extra["correlation_id"] = ctx_correlation_id
        self._ensure_operation_and_status(extra, logging.WARNING)
        kwargs_dict["extra"] = extra
        self.logger.warning(msg, *args, **kwargs_dict)

    def error(self, msg: object, *args: object, **kwargs: object) -> None:
        """Log an error message with structured fields."""
        kwargs_dict = cast(dict[str, Any], kwargs)
        extra = kwargs_dict.get("extra", {})
        if not isinstance(extra, dict):
            extra = {}
        # Merge self.extra (fields from LoggerAdapter constructor) into extra
        if isinstance(self.extra, dict):
            for key, value in self.extra.items():
                if key not in extra:
                    extra[key] = value
        # Inject correlation_id from context if not provided
        if "correlation_id" not in extra:
            ctx_correlation_id = _correlation_id.get()
            if ctx_correlation_id is not None:
                extra["correlation_id"] = ctx_correlation_id
        self._ensure_operation_and_status(extra, logging.ERROR)
        kwargs_dict["extra"] = extra
        self.logger.error(msg, *args, **kwargs_dict)

    def critical(self, msg: object, *args: object, **kwargs: object) -> None:
        """Log a critical message with structured fields."""
        kwargs_dict = cast(dict[str, Any], kwargs)
        extra = kwargs_dict.get("extra", {})
        if not isinstance(extra, dict):
            extra = {}
        # Merge self.extra (fields from LoggerAdapter constructor) into extra
        if isinstance(self.extra, dict):
            for key, value in self.extra.items():
                if key not in extra:
                    extra[key] = value
        # Inject correlation_id from context if not provided
        if "correlation_id" not in extra:
            ctx_correlation_id = _correlation_id.get()
            if ctx_correlation_id is not None:
                extra["correlation_id"] = ctx_correlation_id
        self._ensure_operation_and_status(extra, logging.CRITICAL)
        kwargs_dict["extra"] = extra
        self.logger.critical(msg, *args, **kwargs_dict)

    def log(self, level: int, msg: object, *args: object, **kwargs: object) -> None:
        """Log a message at the given level with structured fields."""
        kwargs_dict = cast(dict[str, Any], kwargs)
        extra = kwargs_dict.get("extra", {})
        if not isinstance(extra, dict):
            extra = {}
        # Merge self.extra (fields from LoggerAdapter constructor) into extra
        if isinstance(self.extra, dict):
            for key, value in self.extra.items():
                if key not in extra:
                    extra[key] = value
        # Inject correlation_id from context if not provided
        if "correlation_id" not in extra:
            ctx_correlation_id = _correlation_id.get()
            if ctx_correlation_id is not None:
                extra["correlation_id"] = ctx_correlation_id
        self._ensure_operation_and_status(extra, level)
        kwargs_dict["extra"] = extra
        self.logger.log(level, msg, *args, **kwargs_dict)

    def log_success(
        self,
        message: str,
        *,
        operation: str | None = None,
        duration_ms: float | None = None,
        **fields: object,
    ) -> None:
        """Log a successful operation with structured fields.

        Parameters
        ----------
        message : str
            Success message.
        operation : str | None, optional
            Operation name. If not provided, uses current context value or "unknown".
            Defaults to ``None``.
        duration_ms : float | None, optional
            Operation duration in milliseconds. Defaults to ``None``.
        **fields : object
            Additional structured fields to include in log record.

        Examples
        --------
        >>> from kgfoundry_common.logging import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.log_success("Index built", operation="build_index", duration_ms=1234.5)
        """
        extra: dict[str, object] = {"status": "success"}
        if operation is not None:
            extra["operation"] = operation
        if duration_ms is not None:
            extra["duration_ms"] = duration_ms
        extra.update(fields)
        self.info(message, extra=extra)

    def log_failure(
        self,
        message: str,
        *,
        exception: Exception | None = None,
        operation: str | None = None,
        duration_ms: float | None = None,
        **fields: object,
    ) -> None:
        """Log a failure with structured fields and optional exception chaining.

        Parameters
        ----------
        message : str
            Failure message.
        exception : Exception | None, optional
            Exception that caused the failure (preserved in extras). Defaults to ``None``.
        operation : str | None, optional
            Operation name. Defaults to ``None``.
        duration_ms : float | None, optional
            Operation duration in milliseconds. Defaults to ``None``.
        **fields : object
            Additional structured fields.

        Examples
        --------
        >>> from kgfoundry_common.logging import get_logger
        >>> logger = get_logger(__name__)
        >>> try:
        ...     raise ValueError("Invalid data")
        ... except ValueError as e:
        ...     logger.log_failure("Failed to process", exception=e, operation="process_data")
        """
        extra: dict[str, object] = {"status": "error"}
        if operation is not None:
            extra["operation"] = operation
        if duration_ms is not None:
            extra["duration_ms"] = duration_ms
        if exception is not None:
            extra["error_type"] = exception.__class__.__name__
            extra["error_detail"] = str(exception)
        extra.update(fields)
        self.error(message, extra=extra)

    def log_io(
        self,
        message: str,
        *,
        operation: str | None = None,
        io_type: str = "unknown",
        size_bytes: int | None = None,
        duration_ms: float | None = None,
        **fields: object,
    ) -> None:
        """Log I/O operation with structured fields.

        Parameters
        ----------
        message : str
            I/O operation message.
        operation : str | None, optional
            Operation name. Defaults to ``None``.
        io_type : str, optional
            I/O type ("read", "write", "delete", "unknown"). Defaults to ``"unknown"``.
        size_bytes : int | None, optional
            Bytes transferred/processed. Defaults to ``None``.
        duration_ms : float | None, optional
            I/O duration in milliseconds. Defaults to ``None``.
        **fields : object
            Additional structured fields.

        Examples
        --------
        >>> from kgfoundry_common.logging import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.log_io(
        ...     "Downloaded file",
        ...     operation="download",
        ...     io_type="read",
        ...     size_bytes=1024,
        ...     duration_ms=500.0,
        ... )
        """
        extra: dict[str, object] = {"status": "success", "io_type": io_type}
        if operation is not None:
            extra["operation"] = operation
        if size_bytes is not None:
            extra["size_bytes"] = size_bytes
        if duration_ms is not None:
            extra["duration_ms"] = duration_ms
        extra.update(fields)
        self.info(message, extra=extra)


def get_logger(name: str) -> LoggerAdapter:
    """Get a logger adapter with structured logging support.

    <!-- auto:docstring-builder v1 -->

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

    <!-- auto:docstring-builder v1 -->

    This function sets up structured JSON logging to stdout. It should
    be called once at application startup.

    Parameters
    ----------
    level : int, optional
        Logging level (default: logging.INFO).
        Defaults to ``20``.

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

    <!-- auto:docstring-builder v1 -->

    This function uses `contextvars.ContextVar` to ensure correlation IDs
    propagate correctly through async tasks and thread pools without
    cross-contamination between concurrent requests.

    Parameters
    ----------
    correlation_id : str | NoneType
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

    <!-- auto:docstring-builder v1 -->

    Returns
    -------
    str | NoneType
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

    <!-- auto:docstring-builder v1 -->

    This class manages correlation ID context using `contextvars.ContextVar`,
    ensuring IDs propagate correctly through async tasks and thread pools
    without cross-contamination between concurrent requests.

    Parameters
    ----------
    correlation_id : str | None
        Correlation ID to set in context.

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
        correlation_id : str | NoneType
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

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit correlation context and restore previous correlation ID.

        Parameters
        ----------
        exc_type : type[BaseException] | None
            Exception type (if any).
        exc_val : BaseException | None
            Exception value (if any).
        exc_tb : object
            Exception traceback (if any).
        """
        if self._token is not None:
            _correlation_id.reset(self._token)


class _WithFieldsContext(AbstractContextManager[LoggerAdapter]):
    """Context manager implementation for `with_fields`."""

    def __init__(
        self, logger: logging.Logger | LoggerAdapter, fields: Mapping[str, object]
    ) -> None:
        self._logger = logger
        self._fields = dict(fields)
        self._token: contextvars.Token[str | None] | None = None
        self._adapter: LoggerAdapter | None = None

    def __enter__(self) -> LoggerAdapter:
        base_logger = (
            self._logger.logger if isinstance(self._logger, LoggerAdapter) else self._logger
        )
        correlation_id = self._fields.get("correlation_id")
        if isinstance(correlation_id, str):
            self._token = _correlation_id.set(correlation_id)
        self._adapter = LoggerAdapter(base_logger, dict(self._fields))
        return self._adapter

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        if self._token is not None:
            _correlation_id.reset(self._token)
        return None


def with_fields(
    logger: logging.Logger | LoggerAdapter,
    **fields: object,
) -> AbstractContextManager[LoggerAdapter]:
    """Context manager for attaching structured fields to log entries.

    <!-- auto:docstring-builder v1 -->

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

    Returns
    -------
    ContextManager[LoggerAdapter]
        Logger adapter with bound fields and correlation_id in context.
    """
    return _WithFieldsContext(logger, fields)


def measure_duration() -> tuple[float, float]:
    """Get current monotonic time for measuring operation duration.

    Returns
    -------
    tuple[float, float]
        Tuple of (start_time, current_time) both from monotonic clock.

    Examples
    --------
    >>> start_time, _ = measure_duration()
    >>> # do work...
    >>> _, end_time = measure_duration()
    >>> duration_ms = (end_time - start_time) * 1000
    """
    current = time.monotonic()
    return current, current
