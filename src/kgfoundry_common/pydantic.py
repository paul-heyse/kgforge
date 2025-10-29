if TYPE_CHECKING:
    # [nav:anchor BaseModel]
    class BaseModel:
        """Structural subset of :class:`pydantic.BaseModel` exposed for mypy.
<!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        **data : object
            Keyword arguments accepted by the real ``BaseModel`` constructor.

        Attributes
        ----------
        model_config : ClassVar[object]
            Placeholder for the runtime model configuration object.
        """

        model_config: ClassVar[object]

        def __init__(self, **data: object) -> None:
            raise NotImplementedError

        @classmethod
        def model_validate(cls, obj: object, /) -> Self:
            """Return a validated model instance from arbitrary input.
<!-- auto:docstring-builder v1 -->

            Parameters
            ----------
            obj : object
                Instance to validate against the model schema.

            Returns
            -------
            Self
                Validated model instance.
            """

            raise NotImplementedError

        def model_dump(self, *args: object, **kwargs: object) -> dict[str, object]:
            """Serialize the model into a dictionary representation.
<!-- auto:docstring-builder v1 -->

            Parameters
            ----------
            *args : object
                Positional arguments forwarded to ``pydantic.BaseModel.model_dump``.
            **kwargs : object
                Keyword arguments forwarded to ``pydantic.BaseModel.model_dump``.

            Returns
            -------
            dict[str, object]
                Dictionary representation produced by Pydantic.
            """

            raise NotImplementedError

        def model_dump_json(self, *args: object, **kwargs: object) -> str:
            """Serialize the model into a JSON string.
<!-- auto:docstring-builder v1 -->

            Parameters
            ----------
            *args : object
                Positional arguments forwarded to ``pydantic.BaseModel.model_dump_json``.
            **kwargs : object
                Keyword arguments forwarded to ``pydantic.BaseModel.model_dump_json``.

            Returns
            -------
            str
                JSON representation produced by Pydantic.
            """

            raise NotImplementedError
else:
    from pydantic import BaseModel as _PydanticBaseModel

    BaseModel = _PydanticBaseModel
