from __future__ import annotations

from typing import Any, TypeVar

T = TypeVar("T")

def replace(obj: T, /, **changes: Any) -> T: ...

from __future__ import annotations

def replace[T](instance: T, **changes: object) -> T: ...

__all__: tuple[str, ...]
