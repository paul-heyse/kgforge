from __future__ import annotations

import logging
from collections.abc import Mapping, MutableMapping
from contextlib import AbstractContextManager
from typing import Any, TypedDict

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

class LogContextExtra(TypedDict, total=False):
    correlation_id: str
    operation: str
    status: str
    duration_ms: float
    service: str
    endpoint: str

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str: ...

class LoggerAdapter(logging.LoggerAdapter[logging.Logger, MutableMapping[str, object]]):
    logger: logging.Logger
    extra: MutableMapping[str, object]

    def process(self, msg: str, kwargs: Mapping[str, Any]) -> tuple[str, Any]: ...
    def debug(self, msg: object, *args: object, **kwargs: object) -> None: ...
    def info(self, msg: object, *args: object, **kwargs: object) -> None: ...
    def warning(self, msg: object, *args: object, **kwargs: object) -> None: ...
    def error(self, msg: object, *args: object, **kwargs: object) -> None: ...
    def critical(self, msg: object, *args: object, **kwargs: object) -> None: ...
    def log(self, level: int, msg: object, *args: object, **kwargs: object) -> None: ...
    def log_success(
        self,
        message: str,
        *,
        operation: str | None = ...,
        duration_ms: float | None = ...,
        **fields: object,
    ) -> None: ...
    def log_failure(
        self,
        message: str,
        *,
        exception: Exception | None = ...,
        operation: str | None = ...,
        duration_ms: float | None = ...,
        **fields: object,
    ) -> None: ...
    def log_io(
        self,
        message: str,
        *,
        operation: str | None = ...,
        io_type: str = ...,
        size_bytes: int | None = ...,
        duration_ms: float | None = ...,
        **fields: object,
    ) -> None: ...

def get_logger(name: str) -> LoggerAdapter: ...
def setup_logging(level: int = ...) -> None: ...
def set_correlation_id(correlation_id: str | None) -> None: ...
def get_correlation_id() -> str | None: ...

class CorrelationContext(AbstractContextManager[CorrelationContext]):
    def __init__(self, correlation_id: str | None) -> None: ...
    def __enter__(self) -> CorrelationContext: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None: ...

def with_fields(
    logger: logging.Logger | LoggerAdapter,
    **fields: object,
) -> AbstractContextManager[LoggerAdapter]: ...
def measure_duration() -> tuple[float, float]: ...
