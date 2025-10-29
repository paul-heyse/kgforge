"""Typed wrappers around Pydantic primitives used throughout the project."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Self

__all__ = ["BaseModel"]

__navmap__ = {
    "exports": ["BaseModel"],
    "sections": [
        {
            "id": "public-api",
            "symbols": ["BaseModel"],
        },
    ],
    "module_meta": {
        "owner": "@kgfoundry-common",
        "stability": "stable",
        "since": "0.1.0",
    },
    "synopsis": "Compatibility shim providing a pydantic BaseModel alias.",
    "symbols": {
        "BaseModel": {
            "owner": "@kgfoundry-common",
            "stability": "stable",
            "since": "0.1.0",
        },
    },
}

# [nav:section public-api]


if TYPE_CHECKING:
    # [nav:anchor BaseModel]
    class BaseModel:
        """Structural subset of :class:`pydantic.BaseModel` exposed for mypy."""

        model_config: ClassVar[object]

        def __init__(self, **data: object) -> None:
            """Initialize the model with flexible keyword arguments."""
            ...

        @classmethod
        def model_validate(cls, obj: object, /) -> Self:
            """Return a validated model instance from arbitrary input."""
            ...

        def model_dump(self, *args: object, **kwargs: object) -> dict[str, object]:
            """Serialize the model into a dictionary representation."""
            ...

        def model_dump_json(self, *args: object, **kwargs: object) -> str:
            """Serialize the model into a JSON string."""
            ...

else:
    from pydantic import BaseModel as _PydanticBaseModel

    BaseModel = _PydanticBaseModel
