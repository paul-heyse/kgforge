from collections.abc import Callable, Iterable
from typing import Any, TypeVar, dataclass_transform, overload

T = TypeVar("T")

@overload
def field(*, default: T, name: str | None = None) -> T: ...
@overload
def field(*, default_factory: Callable[[], T], name: str | None = None) -> T: ...
@overload
def field(*, name: str | None = None) -> Any: ...
@dataclass_transform(field_specifiers=(field,))
class Struct:
    __struct_fields__: tuple[str, ...]
    __struct_config__: Any
    __match_args__: tuple[str, ...]

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def __init_subclass__(
        cls,
        *,
        kw_only: bool = False,
        **kwargs: Any,
    ) -> None: ...

class MsgspecError(Exception): ...
class DecodeError(MsgspecError): ...

class _JsonModule:
    @overload
    def decode(self, data: bytes, *, type: type[T]) -> T: ...
    @overload
    def decode(self, data: bytes, *, type: None = ...) -> Any: ...
    def decode(self, data: bytes, *, type: type[Any] | None = ...) -> Any: ...

json: _JsonModule

def to_builtins(
    obj: Any,
    *,
    str_keys: bool = False,
    builtin_types: Iterable[type] | None = None,
    enc_hook: Callable[[Any], Any] | None = None,
    order: str | None = None,
) -> Any: ...
