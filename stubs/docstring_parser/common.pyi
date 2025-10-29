from __future__ import annotations

from enum import Enum

class DocstringStyle(Enum):
    REST = ...
    GOOGLE = ...
    NUMPYDOC = ...
    EPYDOC = ...
    AUTO = ...

class DocstringMeta:
    args: list[str]
    description: str | None

    def __init__(self, args: list[str], description: str | None) -> None: ...

class DocstringParam(DocstringMeta):
    arg_name: str
    type_name: str | None
    is_optional: bool | None
    default: str | None

    def __init__(
        self,
        args: list[str],
        description: str | None,
        arg_name: str,
        type_name: str | None,
        is_optional: bool | None,
        default: str | None,
    ) -> None: ...

class DocstringReturns(DocstringMeta):
    type_name: str | None
    is_generator: bool
    return_name: str | None

    def __init__(
        self,
        args: list[str],
        description: str | None,
        type_name: str | None,
        is_generator: bool,
        return_name: str | None = ...,
    ) -> None: ...

class DocstringYields(DocstringReturns):
    pass

class DocstringRaises(DocstringMeta):
    type_name: str | None

    def __init__(
        self,
        args: list[str],
        description: str | None,
        type_name: str | None,
    ) -> None: ...

class DocstringDeprecated(DocstringMeta):
    version: str | None

    def __init__(
        self,
        args: list[str],
        description: str | None,
        version: str | None,
    ) -> None: ...

class DocstringExample(DocstringMeta):
    snippet: str | None

    def __init__(
        self,
        args: list[str],
        snippet: str | None,
        description: str | None,
    ) -> None: ...

class Docstring:
    short_description: str | None
    long_description: str | None
    blank_after_short_description: bool
    blank_after_long_description: bool
    meta: list[DocstringMeta]
    style: DocstringStyle | None

    def __init__(self, style: DocstringStyle | None = ...) -> None: ...
    @property
    def description(self) -> str | None: ...
    @property
    def params(self) -> list[DocstringParam]: ...
    @property
    def raises(self) -> list[DocstringRaises]: ...
    @property
    def returns(self) -> DocstringReturns | None: ...
