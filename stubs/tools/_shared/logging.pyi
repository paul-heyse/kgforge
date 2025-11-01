from __future__ import annotations

import logging
from typing import Any

from kgfoundry_common.logging import LoggerAdapter

LogValue = Any

def get_logger(name: str) -> LoggerAdapter: ...
def with_fields(logger: logging.Logger | LoggerAdapter, **fields: object) -> LoggerAdapter: ...

StructuredLoggerAdapter = LoggerAdapter
