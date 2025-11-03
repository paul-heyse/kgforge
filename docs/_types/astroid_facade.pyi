from __future__ import annotations

from types import ModuleType
from typing import Protocol, runtime_checkable

__all__ = [
    "AstroidBuilderProtocol",
    "AstroidManagerProtocol",
    "coerce_astroid_builder_class",
    "coerce_astroid_manager_class",
]

@runtime_checkable
class AstroidManagerProtocol(Protocol):
    def build_from_file(self, path: str) -> object: ...

@runtime_checkable
class AstroidBuilderProtocol(Protocol):
    def __init__(self, manager: AstroidManagerProtocol | None = None) -> None: ...
    def file_build(self, file_path: str, module_name: str) -> object: ...

def coerce_astroid_manager_class(module: ModuleType) -> type[AstroidManagerProtocol]: ...
def coerce_astroid_builder_class(module: ModuleType) -> type[AstroidBuilderProtocol]: ...
