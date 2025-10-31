from __future__ import annotations

from collections.abc import Callable

class Struct:
    def __init__(self, *args: object, **kwargs: object) -> None: ...
    def __init_subclass__(cls, *, kw_only: bool | None = None) -> None: ...

class UnsetType: ...

UNSET: UnsetType

def field(
    *,
    default: object | None = None,
    default_factory: Callable[[], object] | None = None,
) -> object: ...
def to_builtins(obj: object) -> object: ...

__all__: tuple[str, ...]
