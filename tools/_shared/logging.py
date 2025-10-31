"""Logging helpers shared across tooling packages."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

LogValue = Any


class StructuredLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that merges structured ``extra`` fields."""

    def __init__(
        self, logger: logging.Logger, *, extra: Mapping[str, LogValue] | None = None
    ) -> None:
        super().__init__(logger, dict(extra or {}))

    def bind(self, **fields: LogValue) -> StructuredLoggerAdapter:
        """Return a new adapter with ``fields`` merged into ``extra``."""
        merged = dict(self.extra)
        merged.update(fields)
        return StructuredLoggerAdapter(self.logger, extra=merged)

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        extra = dict(self.extra)
        passed = kwargs.get("extra")
        if isinstance(passed, Mapping):
            extra.update(passed)
        kwargs["extra"] = extra
        return msg, kwargs


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured with a ``NullHandler``."""
    logger = logging.getLogger(name)
    if all(not isinstance(handler, logging.NullHandler) for handler in logger.handlers):
        logger.addHandler(logging.NullHandler())
    return logger


def with_fields(logger: logging.Logger, **fields: LogValue) -> StructuredLoggerAdapter:
    """Return a structured adapter bound to ``fields``."""
    return StructuredLoggerAdapter(logger, extra=fields)


__all__ = ["LogValue", "StructuredLoggerAdapter", "get_logger", "with_fields"]
