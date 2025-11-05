"""Compatibility helpers for Pydantic adapters used in kgfoundry."""
# [nav:section public-api]

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Self

if TYPE_CHECKING:
    # [nav:anchor BaseModel]
    class BaseModel:
        """Typing-friendly stub that mirrors Pydantic's ``BaseModel``.

        Populates the model from keyword arguments.

        Parameters
        ----------
        **data : Any
            Keyword arguments accepted by the Pydantic model.

        Attributes
        ----------
        model_config : ClassVar[object]
            Pydantic model configuration dictionary.
        """

        model_config: ClassVar[object]

        def __init__(self, **data: object) -> None:
            raise NotImplementedError

        @classmethod
        def model_validate(cls, obj: object, /) -> Self:
            """Validate ``obj`` using the underlying Pydantic implementation.

            Parameters
            ----------
            obj : Any
                Instance or mapping to validate.
            strict : bool | None, optional
                Whether to forbid coercion during validation.
                Defaults to ``None`` (defer to Pydantic).
            extra : ExtraValues | None, optional
                Strategy for handling extra keys.
                Defaults to ``None`` (use model configuration).
            from_attributes : bool | None, optional
                Allow attribute-based population when ``obj`` is not a mapping.
                Defaults to ``None`` (follow model configuration).
            context : Any | None, optional
                Context data available to validators.
                Defaults to ``None``.
            by_alias : bool | None, optional
                Interpret alias names when reading ``obj``.
                Defaults to ``None`` (inherit from configuration).
            by_name : bool | None, optional
                Permit field-name based population alongside aliases.
                Defaults to ``None``.
            """
            raise NotImplementedError

        def model_dump(self, **model_dump_kwargs: object) -> dict[str, object]:
            """Return the dictionary representation produced by Pydantic.

            Parameters
            ----------
            **model_dump_kwargs : dict[str, object]
                Keyword arguments forwarded to :meth:`pydantic.BaseModel.model_dump`.
            """
            del self, model_dump_kwargs
            raise NotImplementedError

        def model_dump_json(self, **model_dump_json_kwargs: object) -> str:
            """Return the JSON representation produced by Pydantic.

            Parameters
            ----------
            **model_dump_json_kwargs : dict[str, object]
                Keyword arguments forwarded to :meth:`pydantic.BaseModel.model_dump_json`.
            """
            del self, model_dump_json_kwargs
            raise NotImplementedError

else:
    from pydantic import BaseModel as _PydanticBaseModel

    BaseModel = _PydanticBaseModel
