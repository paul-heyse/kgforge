"""Type stubs for internal namespace proxy module."""

from __future__ import annotations

from collections.abc import Callable, Iterable, MutableMapping
from dataclasses import dataclass
from types import ModuleType
from typing import TypeVar

T = TypeVar("T")

@dataclass(slots=True)
class NamespaceRegistry:
    """Typed registry for lazy-loading module symbols."""

    _registry: dict[str, Callable[[], object]]
    _cache: dict[str, object]

    def __init__(self) -> None: ...
    def register(self, name: str, loader: Callable[[], T]) -> None: ...
    def resolve(self, name: str) -> object: ...
    def list_symbols(self) -> list[str]: ...

def namespace_getattr(module: ModuleType, name: str) -> object: ...
def namespace_exports(module: ModuleType) -> list[str]: ...
def namespace_attach(
    module: ModuleType,
    target: MutableMapping[str, object],
    names: Iterable[str],
) -> None: ...
def namespace_dir(module: ModuleType, exports: Iterable[str]) -> list[str]: ...
