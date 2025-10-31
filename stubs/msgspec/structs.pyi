from __future__ import annotations

from typing import TypeVar

_T = TypeVar("_T")

def replace(instance: _T, **changes: object) -> _T: ...

__all__: tuple[str, ...]
