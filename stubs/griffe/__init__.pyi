from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Forward declarations - matched to runtime Griffe 1.14.0
class Object:
    path: str
    canonical_path: str | None
    name: str
    lineno: int | None
    endlineno: int | None
    docstring: Docstring | None
    kind: str
    members: dict[str, Object]
    is_alias: bool
    is_async: bool | None
    is_property: bool | None
    inherited: bool
    private: bool

    def __init__(self) -> None: ...

class Alias(Object):
    target: str | Object | None

class Module(Object): ...
class Package: ...
class Class(Object): ...
class Function(Object): ...
class Attribute(Object): ...
class TypeAlias(Object): ...

class Docstring:
    value: str
    lineno: int | None
    endlineno: int | None

# GriffeLoader - main loader class
class GriffeLoader:
    def __init__(
        self,
        *,
        search_paths: Sequence[str | Path] | None = None,
        allow_inspection: bool = True,
        force_inspection: bool = False,
        docstring_parser: Any = None,  # noqa: ANN401
        docstring_options: dict[str, Any] | None = None,
    ) -> None: ...
    def load(
        self,
        objspec: str | Path | None = None,
        /,
        *,
        submodules: bool = True,
        try_relative_path: bool = True,
        find_stubs_package: bool = False,
    ) -> Object | Alias: ...

# Module-level load function
def load(
    objspec: str | Path | None = None,
    /,
    *,
    submodules: bool = True,
    try_relative_path: bool = True,
    extensions: Any = None,  # noqa: ANN401
    search_paths: Sequence[str | Path] | None = None,
    docstring_parser: Any = None,  # noqa: ANN401
    docstring_options: Any = None,  # noqa: ANN401
    allow_inspection: bool = True,
    force_inspection: bool = False,
    store_source: bool = True,
    find_stubs_package: bool = False,
    resolve_aliases: bool = False,
    resolve_external: bool | None = None,
    resolve_implicit: bool = False,
) -> Object | Alias: ...

# Exceptions from griffe module
class GriffeError(Exception): ...
class LoadingError(GriffeError): ...
class NameResolutionError(GriffeError): ...
class AliasResolutionError(GriffeError): ...
class CyclicAliasError(GriffeError): ...
class UnimportableModuleError(GriffeError): ...
class BuiltinModuleError(GriffeError): ...
class ExtensionError(GriffeError): ...
class ExtensionNotLoadedError(ExtensionError): ...

__all__ = [
    "Alias",
    "Attribute",
    "Class",
    "Docstring",
    "Function",
    "GriffeError",
    "GriffeLoader",
    "LoadingError",
    "Module",
    "Object",
    "Package",
    "TypeAlias",
    "load",
]
