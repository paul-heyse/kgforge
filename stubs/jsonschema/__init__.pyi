from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence

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

class Draft202012Validator:
    schema: Mapping[str, object]

    def __init__(self, schema: Mapping[str, object]) -> None: ...
    def validate(self, instance: object) -> None: ...
    @classmethod
    def check_schema(cls, schema: Mapping[str, object]) -> None: ...
    def iter_errors(self, instance: object) -> Iterable[ValidationError]: ...

def validate(instance: object, schema: Mapping[str, object]) -> None: ...
