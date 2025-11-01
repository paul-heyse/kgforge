from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

class Struct:
    """Lightweight structural base class used for msgspec models."""

    def __init_subclass__(
        cls,
        *,
        array_like: bool | None = None,
        frozen: bool | None = None,
        gc: bool | None = None,
        kw_only: bool | None = None,
    ) -> None: ...

class UnsetType:
    """Sentinel for omitted struct fields."""

UNSET: UnsetType

def to_builtins(obj: Any, *, str_keys: bool | None = None) -> Any: ...
def field(
    *,
    default: Any = ...,
    default_factory: Callable[[], Any] | None = ...,
    name: str | None = ...,
    alias: str | None = ...,
) -> Any: ...

class _StructsModule:
    def replace(self, obj: T, /, **changes: Any) -> T: ...

structs: _StructsModule

class _JsonModule:
    def encode(self, obj: Any, *, enc_hook: Any | None = None) -> bytes: ...
    def decode(
        self, data: bytes | bytearray | memoryview | str, *, type: type[T] | None = None
    ) -> T: ...
    def schema(self, obj: Any) -> Any: ...

json: _JsonModule

class StructError(Exception): ...
class DecodeError(StructError): ...
class ValidationError(StructError): ...

def convert(obj: Any, *, type: type[T] | None = ...) -> T: ...
