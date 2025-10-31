from __future__ import annotations

from collections import deque
from collections.abc import Sequence

class _Error(Exception):
    """Base class for jsonschema errors."""
    message: str

class ValidationError(_Error):
    """Exception raised when validation fails."""
    absolute_path: deque[object]
    absolute_schema_path: deque[object]
    path: Sequence[object]
    message: str

class SchemaError(_Error):
    """Exception raised when schema is invalid."""
    message: str

