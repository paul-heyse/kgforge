from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence

__all__ = ["Draft202012Validator", "RefResolver", "SchemaError", "ValidationError", "validate"]

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

class RefResolver:
    """JSON Schema reference resolver."""

    def __init__(
        self,
        base_uri: str = "",
        referrer: Mapping[str, object] | None = None,
        store: dict[str, Mapping[str, object]] | None = None,
    ) -> None:
        """Initialize reference resolver.

        Parameters
        ----------
        base_uri : str, optional
            Base URI for resolving relative references.
            Defaults to "".
        referrer : Mapping[str, object] | None, optional
            Referrer schema for resolving relative references.
            Defaults to None.
        store : dict[str, Mapping[str, object]] | None, optional
            Store of resolved schemas.
            Defaults to None.
        """
        ...

    @classmethod
    def from_schema(
        cls,
        schema: Mapping[str, object],
        store: dict[str, Mapping[str, object]] | None = None,
    ) -> RefResolver:
        """Create resolver from schema."""
        ...

class Draft202012Validator:
    """JSON Schema Draft 2020-12 validator."""

    schema: Mapping[str, object]

    def __init__(
        self,
        schema: Mapping[str, object],
        resolver: RefResolver | None = None,
        format_checker: object | None = None,  # FormatChecker type unknown
    ) -> None:
        """Initialize Draft 2020-12 validator.

        Parameters
        ----------
        schema : Mapping[str, object]
            JSON Schema to validate against.
        resolver : RefResolver | None, optional
            Reference resolver for $ref resolution.
            Defaults to None.
        format_checker : Any | None, optional
            Format checker for format validation.
            Defaults to None.
        """
        ...

    def validate(self, instance: object) -> None:
        """Validate instance against schema.

        Parameters
        ----------
        instance : object
            Instance to validate.

        Raises
        ------
        ValidationError
            If validation fails.
        """
        ...

    @classmethod
    def check_schema(cls, schema: Mapping[str, object]) -> None:
        """Check schema validity."""
        ...

    def iter_errors(self, instance: object) -> Iterable[ValidationError]:
        """Iterate over validation errors."""
        ...

def validate(instance: object, schema: Mapping[str, object]) -> None:
    """Validate instance against schema."""
    ...
