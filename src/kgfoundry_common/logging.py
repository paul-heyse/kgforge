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
from typing import Any, Final

from kgfoundry_common.navmap_types import NavMap

__all__ = ["JsonFormatter", "LoggerAdapter", "get_logger", "setup_logging"]

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

        # Extract structured fields from extra
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

        # Add any additional extra fields
        for key in ("run_id", "doc_id", "chunk_id", "trace_id", "span_id"):
            value = getattr(record, key, None)
            if value is not None:
                data[key] = value

        return json.dumps(data, default=str)


class LoggerAdapter(logging.LoggerAdapter[logging.Logger]):
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

    def process(self, msg: str, kwargs: Any) -> tuple[str, Any]:  # type: ignore[override]  # noqa: ANN401  # LoggerAdapter uses Any
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

        # Inject correlation_id from context if not provided
        if "correlation_id" not in extra:
            ctx_correlation_id = _correlation_id.get()
            if ctx_correlation_id is not None:
                extra["correlation_id"] = ctx_correlation_id

        # Ensure operation and status are present (defaults if missing)
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

        return msg, kwargs


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
