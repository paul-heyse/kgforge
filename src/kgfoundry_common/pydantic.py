"""Compatibility helpers for Pydantic adapters used in kgfoundry."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar, Self

if TYPE_CHECKING:
    # [nav:anchor BaseModel]
    class BaseModel:
        """Typing-friendly stub that mirrors Pydantic's ``BaseModel``."""

        model_config: ClassVar[object]

        def __init__(self, **data: object) -> None:
            """Populate the model from keyword arguments."""
            raise NotImplementedError

        @classmethod
        def model_validate(cls, obj: object, /) -> Self:
            """Validate ``obj`` using the underlying Pydantic implementation."""
            raise NotImplementedError

        def model_dump(  # noqa: PLR0913
            self,
            *,
            mode: str = "python",
            include: object | None = None,
            exclude: object | None = None,
            context: object | None = None,
            by_alias: bool | None = None,
            exclude_unset: bool = False,
            exclude_defaults: bool = False,
            exclude_none: bool = False,
            exclude_computed_fields: bool = False,
            round_trip: bool = False,
            warnings: bool | str = True,
            fallback: Callable[[object], object] | None = None,
            serialize_as_any: bool = False,
        ) -> dict[str, object]:
            """Return the dictionary representation produced by Pydantic."""
            raise NotImplementedError

        def model_dump_json(  # noqa: PLR0913
            self,
            *,
            indent: int | None = None,
            ensure_ascii: bool = False,
            include: object | None = None,
            exclude: object | None = None,
            context: object | None = None,
            by_alias: bool | None = None,
            exclude_unset: bool = False,
            exclude_defaults: bool = False,
            exclude_none: bool = False,
            exclude_computed_fields: bool = False,
            round_trip: bool = False,
            warnings: bool | str = True,
            fallback: Callable[[object], object] | None = None,
            serialize_as_any: bool = False,
        ) -> str:
            """Return the JSON representation produced by Pydantic."""
            raise NotImplementedError

else:
    from pydantic import BaseModel as _PydanticBaseModel

    BaseModel = _PydanticBaseModel
