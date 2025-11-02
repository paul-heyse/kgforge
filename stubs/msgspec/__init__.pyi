from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar, overload

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

def to_builtins(obj: object, *, str_keys: bool | None = None) -> object: ...
def field(
    *,
    default: object = ...,
    default_factory: Callable[[], object] | None = ...,
    name: str | None = ...,
    alias: str | None = ...,
) -> object: ...
@overload
def struct(
    cls: type[T],
    *,
    array_like: bool | None = None,
    frozen: bool | None = None,
    gc: bool | None = None,
    kw_only: bool | None = None,
    tag: object = ...,
    tag_field: str | None = ...,
    rename: object = ...,
    omit_defaults: bool | None = None,
    slots: bool | None = None,
) -> type[T]: ...
@overload
def struct(
    *,
    array_like: bool | None = None,
    frozen: bool | None = None,
    gc: bool | None = None,
    kw_only: bool | None = None,
    tag: object = ...,
    tag_field: str | None = ...,
    rename: object = ...,
    omit_defaults: bool | None = None,
    slots: bool | None = None,
) -> Callable[[type[T]], type[T]]: ...

class _StructsModule:
    def replace(self, obj: T, /, **changes: object) -> T: ...

structs: _StructsModule

class _JsonModule:
    def encode(
        self, obj: object, *, enc_hook: Callable[[object], object] | None = None
    ) -> bytes: ...
    def decode(
        self,
        data: bytes | bytearray | memoryview | str,
        *,
        type: type[T] | None = None,  # noqa: A002 - public API uses `type`
    ) -> T: ...
    def schema(self, obj: object) -> object: ...

    class Codec:
        """JSON codec for deterministic serialization."""

        def encode(
            self, obj: object, *, enc_hook: Callable[[object], object] | None = None
        ) -> bytes: ...
        def decode(
            self,
            data: bytes | bytearray | memoryview | str,
            *,
            type: type[T] | None = None,  # noqa: A002 - public API uses `type`
        ) -> T: ...

json: _JsonModule

class StructError(Exception): ...
class DecodeError(StructError): ...
class ValidationError(StructError): ...

def convert[T](obj: object, *, type: type[T] | None = ...) -> T: ...  # noqa: A002 - public API uses `type`
