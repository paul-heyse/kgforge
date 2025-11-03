"""Typed facades for jsonschema usage across the codebase.

This module centralizes imports from :mod:`jsonschema` so that mypy sees
concrete types instead of ``Any``.  The upstream stubs expose several
untyped entry points (e.g. :class:`Draft202012Validator`,
:func:`jsonschema.validate`), so we wrap them with Protocol-based casts and
re-export the typed surfaces for internal use.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol, cast

from jsonschema import validate as _jsonschema_validate
from jsonschema.exceptions import SchemaError as _SchemaError
from jsonschema.exceptions import ValidationError as _ValidationError
from jsonschema.validators import Draft202012Validator as _Draft202012Validator


class ValidationErrorProtocol(Protocol):
    """Typed view over ``jsonschema.exceptions.ValidationError`` instances."""

    message: str
    absolute_path: Sequence[object]
    path: Sequence[object]


class Draft202012ValidatorProtocol(Protocol):
    """Typed facade for :class:`jsonschema.validators.Draft202012Validator`."""

    def __init__(self, schema: Mapping[str, object], *args: object, **kwargs: object) -> None:
        """Initialize the validator with ``schema``."""
        ...

    @classmethod
    def check_schema(cls, schema: Mapping[str, object]) -> None:
        """Validate that ``schema`` conforms to the Draft 2020-12 meta-schema."""
        ...

    def iter_errors(self, instance: object) -> Iterable[ValidationErrorProtocol]:
        """Yield validation errors for ``instance`` without raising."""
        ...

    def validate(self, instance: object) -> None:
        """Validate ``instance`` against the configured schema."""
        ...


Draft202012Validator = cast(type[Draft202012ValidatorProtocol], _Draft202012Validator)
SchemaError = cast(type[Exception], _SchemaError)
ValidationError = cast(type[Exception], _ValidationError)


def validate(instance: object, schema: Mapping[str, object]) -> None:
    """Validate ``instance`` against ``schema`` using jsonschema."""
    _jsonschema_validate(instance=instance, schema=schema)


__all__ = [
    "Draft202012Validator",
    "Draft202012ValidatorProtocol",
    "SchemaError",
    "ValidationError",
    "ValidationErrorProtocol",
    "create_draft202012_validator",
    "validate",
]


def create_draft202012_validator(
    schema: Mapping[str, object],
) -> Draft202012ValidatorProtocol:
    """Return a typed Draft 2020-12 validator for ``schema``."""
    concrete_schema = {str(key): value for key, value in schema.items()}
    instance = _Draft202012Validator(concrete_schema)
    return cast(Draft202012ValidatorProtocol, instance)
