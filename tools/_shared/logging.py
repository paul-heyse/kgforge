"""Logging helpers shared across tooling packages.

This module delegates to kgfoundry_common.logging for structured logging with
correlation IDs and JSON formatting. Libraries remain handler-agnostic; CLIs
configure handlers via setup_logging().

Examples
--------
>>> from tools._shared.logging import get_logger, with_fields
>>> logger = get_logger(__name__)
>>> logger.info("Operation started", extra={"operation": "build", "status": "started"})
>>> # Use with_fields for structured context
>>> adapter = with_fields(logger, correlation_id="req-123", operation="build")
>>> adapter.info("Processing files", extra={"file_count": 10})
"""

from __future__ import annotations

import logging
from typing import Any

from kgfoundry_common.logging import (
    LoggerAdapter,
)
from kgfoundry_common.logging import (
    get_logger as _get_logger_base,
)

__all__ = ["LogValue", "StructuredLoggerAdapter", "get_logger", "with_fields"]

# Type alias for log values (same as Any, but more explicit)
LogValue = Any


def get_logger(name: str) -> LoggerAdapter:
    """Return a logger adapter with structured logging support.

    This function delegates to kgfoundry_common.logging.get_logger, which
    provides structured logging with correlation ID propagation and NullHandler
    for libraries. The logger adapter automatically injects correlation_id,
    operation, and status fields.

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
    >>> from tools._shared.logging import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Operation complete", extra={"operation": "index_build", "status": "success"})
    """
    return _get_logger_base(name)


def with_fields(logger: logging.Logger | LoggerAdapter, **fields: LogValue) -> LoggerAdapter:
    """Return a structured adapter bound to ``fields``.

    This function wraps the logger with a LoggerAdapter that merges the
    provided fields into the extra dict for all log calls. The adapter
    also ensures correlation_id propagation from context variables.

    Parameters
    ----------
    logger : logging.Logger | LoggerAdapter
        Base logger to wrap (may already be an adapter).
    **fields : LogValue
        Structured fields to inject into all log entries.

    Returns
    -------
    LoggerAdapter
        Logger adapter with bound fields.

    Examples
    --------
    >>> from tools._shared.logging import get_logger, with_fields
    >>> logger = get_logger(__name__)
    >>> adapter = with_fields(logger, correlation_id="req-123", operation="build")
    >>> adapter.info("Processing", extra={"file_count": 10})
    """
    # kgfoundry_common.logging.LoggerAdapter accepts extra dict as second argument
    # Extract underlying logger if already wrapped
    base_logger = logger.logger if isinstance(logger, LoggerAdapter) else logger

    # Create adapter with fields in extra dict
    return LoggerAdapter(base_logger, fields)


# Re-export for compatibility
StructuredLoggerAdapter = LoggerAdapter
