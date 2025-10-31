from __future__ import annotations

import logging
from typing import Any

def get_logger(name: str) -> logging.Logger: ...
def with_fields(logger: logging.Logger, **fields: Any) -> logging.Logger: ...
