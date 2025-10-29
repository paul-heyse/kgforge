from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Self

if TYPE_CHECKING:
    # [nav:anchor BaseModel]
    class BaseModel:
        """Describe BaseModel.

<!-- auto:docstring-builder v1 -->

Describe the data structure and how instances collaborate with the surrounding package. Highlight how the class supports nearby modules to guide readers through the codebase.

Parameters
----------
**data : Any
    Describe ``data``.
    

Raises
------
NotImplementedError
Raised when TODO for NotImplementedError.
"""

        model_config: ClassVar[object]

        def __init__(self, **data: object) -> None:
            """Describe   init  .

<!-- auto:docstring-builder v1 -->

Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

Parameters
----------
**data : Any
    Describe ``data``.
    

Raises
------
NotImplementedError
Raised when TODO for NotImplementedError.
"""
            raise NotImplementedError

        @classmethod
        def model_validate(cls, obj: object, /) -> Self:
            """Describe model validate.

<!-- auto:docstring-builder v1 -->

Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

Parameters
----------
obj : Any
    Describe ``obj``.
strict : bool | None, optional
    Describe ``strict``.
    Defaults to ``None``.
extra : ExtraValues | None, optional
    Describe ``extra``.
    Defaults to ``None``.
from_attributes : bool | None, optional
    Describe ``from_attributes``.
    Defaults to ``None``.
context : Any | None, optional
    Describe ``context``.
    Defaults to ``None``.
by_alias : bool | None, optional
    Describe ``by_alias``.
    Defaults to ``None``.
by_name : bool | None, optional
    Describe ``by_name``.
    Defaults to ``None``.
    

Returns
-------
Self
    Describe return value.
    

Raises
------
NotImplementedError
Raised when TODO for NotImplementedError.
"""
            raise NotImplementedError

        def model_dump(self, *args: object, **kwargs: object) -> dict[str, object]:
            """Forward to :meth:`pydantic.BaseModel.model_dump` with full.

<!-- auto:docstring-builder v1 -->

flexibility.

Parameters
----------
*args : object
    Positional arguments forwarded to ``BaseModel.model_dump``.
**kwargs : object
    Keyword arguments forwarded to ``BaseModel.model_dump``.
mode : Literal['json', 'python'] | str, optional
    Describe ``mode``.
    Defaults to ``'python'``.
include : IncEx | None, optional
    Describe ``include``.
    Defaults to ``None``.
exclude : IncEx | None, optional
    Describe ``exclude``.
    Defaults to ``None``.
context : Any | None, optional
    Describe ``context``.
    Defaults to ``None``.
by_alias : bool | None, optional
    Describe ``by_alias``.
    Defaults to ``None``.
exclude_unset : bool, optional
    Describe ``exclude_unset``.
    Defaults to ``False``.
exclude_defaults : bool, optional
    Describe ``exclude_defaults``.
    Defaults to ``False``.
exclude_none : bool, optional
    Describe ``exclude_none``.
    Defaults to ``False``.
exclude_computed_fields : bool, optional
    Describe ``exclude_computed_fields``.
    Defaults to ``False``.
round_trip : bool, optional
    Describe ``round_trip``.
    Defaults to ``False``.
warnings : bool | Literal['none', 'warn', 'error'], optional
    Describe ``warnings``.
    Defaults to ``True``.
fallback : Callable[[Any], Any] | None, optional
    Describe ``fallback``.
    Defaults to ``None``.
serialize_as_any : bool, optional
    Describe ``serialize_as_any``.
    Defaults to ``False``.
    

Returns
-------
dict[str, Any]
    Dictionary representation emitted by Pydantic.
"""
            raise NotImplementedError

        def model_dump_json(self, *args: object, **kwargs: object) -> str:
            """Forward to :meth:`pydantic.BaseModel.model_dump_json`.

<!-- auto:docstring-builder v1 -->

Parameters
----------
*args : object
    Positional arguments forwarded to ``BaseModel.model_dump_json``.
**kwargs : object
    Keyword arguments forwarded to ``BaseModel.model_dump_json``.
indent : int | None, optional
    Describe ``indent``.
    Defaults to ``None``.
ensure_ascii : bool, optional
    Describe ``ensure_ascii``.
    Defaults to ``False``.
include : IncEx | None, optional
    Describe ``include``.
    Defaults to ``None``.
exclude : IncEx | None, optional
    Describe ``exclude``.
    Defaults to ``None``.
context : Any | None, optional
    Describe ``context``.
    Defaults to ``None``.
by_alias : bool | None, optional
    Describe ``by_alias``.
    Defaults to ``None``.
exclude_unset : bool, optional
    Describe ``exclude_unset``.
    Defaults to ``False``.
exclude_defaults : bool, optional
    Describe ``exclude_defaults``.
    Defaults to ``False``.
exclude_none : bool, optional
    Describe ``exclude_none``.
    Defaults to ``False``.
exclude_computed_fields : bool, optional
    Describe ``exclude_computed_fields``.
    Defaults to ``False``.
round_trip : bool, optional
    Describe ``round_trip``.
    Defaults to ``False``.
warnings : bool | Literal['none', 'warn', 'error'], optional
    Describe ``warnings``.
    Defaults to ``True``.
fallback : Callable[[Any], Any] | None, optional
    Describe ``fallback``.
    Defaults to ``None``.
serialize_as_any : bool, optional
    Describe ``serialize_as_any``.
    Defaults to ``False``.
    

Returns
-------
str
    JSON string emitted by Pydantic.
"""
            raise NotImplementedError

else:
    from pydantic import BaseModel as _PydanticBaseModel

    BaseModel = _PydanticBaseModel
