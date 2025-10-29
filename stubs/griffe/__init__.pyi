from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

__all__ = [
    "Class",
    "Docstring",
    "Function",
    "GriffeLoader",
    "Module",
    "Object",
    "Parameter",
]

class Docstring:
    value: str

class Parameter:
    name: str
    kind: object
    annotation: object | None
    default: object | None

class Object:
    name: str
    docstring: Docstring | None
    members: Mapping[str, Object]
    lineno: int | None
    endlineno: int | None
    col_offset: int
    decorators: Iterable[Any] | None
    is_async: bool
    is_generator: bool

class Function(Object):
    parameters: list[Parameter]
    return_annotation: object | None
    returns: object | None

class Class(Object): ...
class Module(Object): ...

class GriffeLoader:
    def __init__(self, search_paths: Sequence[str] | None = ...) -> None: ...
    def load(self, module: str) -> Module: ...
    def load_module(self, module: str) -> Module: ...
