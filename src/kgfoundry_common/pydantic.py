"""Typed wrappers around Pydantic primitives used throughout the project."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Self

__all__ = ["BaseModel"]


if TYPE_CHECKING:

    class BaseModel:
        """Structural subset of :class:`pydantic.BaseModel` exposed for mypy."""

        model_config: ClassVar[object]

        def __init__(self, **data: object) -> None:
            """Initialize the model with flexible keyword arguments."""

        @classmethod
        def model_validate(cls, obj: object, /) -> Self:
            """Return a validated model instance from arbitrary input."""

        def model_dump(self, *args: object, **kwargs: object) -> dict[str, object]:
            """Serialize the model into a dictionary representation."""

        def model_dump_json(self, *args: object, **kwargs: object) -> str:
            """Serialize the model into a JSON string."""

else:
    from pydantic import BaseModel as _PydanticBaseModel

    BaseModel = _PydanticBaseModel
