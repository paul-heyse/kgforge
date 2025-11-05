from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

@runtime_checkable
class Docstring(Protocol):
    value: str | None

class _Kind(Protocol):
    name: str

class Object(Protocol):
    path: str
    kind: str | _Kind
    filepath: Path | None
    lineno: int | None
    endlineno: int | None
    docstring: Docstring | None
    inherited: bool
    members: Mapping[str, Object | Alias]
    inherited_members: Mapping[str, Object | Alias]

@runtime_checkable
class Alias(Object, Protocol):
    final_target: Object | Alias | None
    target_path: str | None

@runtime_checkable
class Module(Object, Protocol):
    members: MutableMapping[str, Object | Alias]

class Extensions(Protocol): ...
class AliasResolutionError(Exception): ...
class BuiltinModuleError(Exception): ...
class CyclicAliasError(Exception): ...

def load_extensions(*names: str) -> Extensions: ...

class GriffeLoader:
    def __init__(
        self,
        *,
        extensions: Extensions | None = ...,
        search_paths: Sequence[str | Path] | None = ...,
        docstring_parser: object | None = ...,
        docstring_options: object | None = ...,
        lines_collection: object | None = ...,
        modules_collection: object | None = ...,
        allow_inspection: bool = ...,
        force_inspection: bool = ...,
        store_source: bool = ...,
    ) -> None: ...
    def load(self, module_name: str) -> Module: ...
    def expand_exports(self, module: Module) -> None: ...
    def expand_wildcards(self, module: Module) -> None: ...
    def resolve_aliases(self, *, implicit: bool = ..., external: bool | None = ...) -> None: ...

__all__ = [
    "Alias",
    "AliasResolutionError",
    "BuiltinModuleError",
    "CyclicAliasError",
    "Docstring",
    "GriffeLoader",
    "Module",
    "Object",
    "load_extensions",
]
