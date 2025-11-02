"""Compatibility helpers for Pydantic adapters used in kgfoundry."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar, Self

if TYPE_CHECKING:
    # [nav:anchor BaseModel]
    class BaseModel:
        """Typing-friendly stub that mirrors Pydantic's ``BaseModel``.

        Parameters
        ----------
        **data : Any
            Keyword arguments accepted by the Pydantic model.
        """

        model_config: ClassVar[object]

        def __init__(self, **data: object) -> None:
            """Populate the model from keyword arguments.

            Parameters
            ----------
            **data : Any
                Keyword arguments forwarded to the underlying Pydantic model
                constructor.
            """
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

            Returns
            -------
            Self
                A validated model instance.
            """
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
            """Return the dictionary representation produced by Pydantic.

            Parameters
            ----------
            mode : Literal['json', 'python'] | str, optional
                Rendering mode for the serialized output.
                Defaults to ``'python'``.
            include : IncEx | None, optional
                Fields to include explicitly.
                Defaults to ``None`` (include everything).
            exclude : IncEx | None, optional
                Fields to omit from the output.
                Defaults to ``None``.
            context : Any | None, optional
                Additional context available to custom serializers.
                Defaults to ``None``.
            by_alias : bool | None, optional
                If ``True``, use field aliases instead of field names.
                Defaults to ``None`` (inherit from configuration).
            exclude_unset : bool, optional
                Omit fields that were not explicitly set.
                Defaults to ``False``.
            exclude_defaults : bool, optional
                Remove fields equal to their default values.
                Defaults to ``False``.
            exclude_none : bool, optional
                Exclude ``None`` values from the output.
                Defaults to ``False``.
            exclude_computed_fields : bool, optional
                Skip fields tagged as computed.
                Defaults to ``False``.
            round_trip : bool, optional
                Preserve types that can survive a full serialization round trip.
                Defaults to ``False``.
            warnings : bool | Literal['none', 'warn', 'error'], optional
                Configure how serialization warnings are surfaced.
                Defaults to ``True`` (emit warnings).
            fallback : Callable[[Any], Any] | None, optional
                Serializer invoked when a value cannot be encoded natively.
                Defaults to ``None``.
            serialize_as_any : bool, optional
                Treat values as ``Any`` to bypass strict type enforcement.
                Defaults to ``False``.

            Returns
            -------
            dict[str, Any]
                Dictionary representation of the model.
            """
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
            """Return the JSON representation produced by Pydantic.

            Parameters
            ----------
            indent : int | None, optional
                Number of spaces used to indent JSON output.
                Defaults to ``None`` (compact output).
            ensure_ascii : bool, optional
                Escape non-ASCII characters when ``True``.
                Defaults to ``False``.
            include : IncEx | None, optional
                Fields to include explicitly in the serialized payload.
                Defaults to ``None`` (include everything).
            exclude : IncEx | None, optional
                Fields to omit from the payload.
                Defaults to ``None``.
            context : Any | None, optional
                Supplemental context available to custom serializers.
                Defaults to ``None``.
            by_alias : bool | None, optional
                Output field aliases instead of field names when ``True``.
                Defaults to ``None`` (inherit from configuration).
            exclude_unset : bool, optional
                Omit fields that were never set on the model instance.
                Defaults to ``False``.
            exclude_defaults : bool, optional
                Remove fields that are set to their default values.
                Defaults to ``False``.
            exclude_none : bool, optional
                Skip ``None`` values in the generated JSON.
                Defaults to ``False``.
            exclude_computed_fields : bool, optional
                Exclude computed fields from the serialized output.
                Defaults to ``False``.
            round_trip : bool, optional
                Preserve Python types that can be reconstructed from JSON.
                Defaults to ``False``.
            warnings : bool | Literal['none', 'warn', 'error'], optional
                Controls how serialization warnings are emitted.
                Defaults to ``True`` (emit warnings).
            fallback : Callable[[Any], Any] | None, optional
                Serializer for values unsupported by default encoders.
                Defaults to ``None``.
            serialize_as_any : bool, optional
                Relax type checking by treating values as ``Any`` during
                serialization.
                Defaults to ``False``.

            Returns
            -------
            str
                JSON string representation of the model.
            """
            raise NotImplementedError

else:
    from pydantic import BaseModel as _PydanticBaseModel

    BaseModel = _PydanticBaseModel
