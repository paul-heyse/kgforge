from __future__ import annotations

from collections.abc import Callable
from typing import Any

class UnsetType: ...

UNSET: UnsetType

def field(
    *,
    default: Any | None = ...,
    default_factory: Callable[[], Any] | None = ...,
    name: str | None = None,
) -> Any: ...

class Struct:
    __struct_fields__: tuple[str, ...]

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def __init_subclass__(cls, **kwargs: Any) -> None: ...

def to_builtins(
    obj: Any,
    *,
    str_keys: bool = False,
    builtin_types: tuple[type[Any], ...] | None = None,
    enc_hook: Callable[[Any], Any] | None = None,
    order: str | None = None,
) -> Any: ...
def convert(obj: Any, type: Any, *, strict: bool = True, from_attributes: bool = False) -> Any: ...

class _StructsModule:
    def replace(self, obj: Any, /, **changes: Any) -> Any: ...

structs: _StructsModule

def json_decode(data: bytes, *, type: Any | None = ...) -> Any: ...
def json_encode(obj: Any) -> bytes: ...

class _JsonModule:
    def decode(self, data: bytes, *, type: Any | None = ...) -> Any: ...
    def encode(self, obj: Any) -> bytes: ...

json: _JsonModule

class MsgspecError(Exception): ...
class DecodeError(MsgspecError): ...
class ValidationError(DecodeError): ...
