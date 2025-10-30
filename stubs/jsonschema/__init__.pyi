from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

class ValidationError(Exception):
    path: Sequence[object]
    message: str

class Draft202012Validator:
    schema: Mapping[str, object]

    def __init__(self, schema: Mapping[str, object]) -> None: ...
    def validate(self, instance: object) -> None: ...
    @classmethod
    def check_schema(cls, schema: Mapping[str, object]) -> None: ...
    def iter_errors(self, instance: object) -> Iterable[ValidationError]: ...

def validate(instance: object, schema: Mapping[str, object]) -> None: ...
