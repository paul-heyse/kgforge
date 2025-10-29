from __future__ import annotations

from enum import Enum
from typing import List, Optional


class DocstringStyle(Enum):
    REST = ...
    GOOGLE = ...
    NUMPYDOC = ...
    EPYDOC = ...
    AUTO = ...


class DocstringMeta:
    args: List[str]
    description: Optional[str]

    def __init__(self, args: List[str], description: Optional[str]) -> None: ...


class DocstringParam(DocstringMeta):
    arg_name: str
    type_name: Optional[str]
    is_optional: Optional[bool]
    default: Optional[str]

    def __init__(
        self,
        args: List[str],
        description: Optional[str],
        arg_name: str,
        type_name: Optional[str],
        is_optional: Optional[bool],
        default: Optional[str],
    ) -> None: ...


class DocstringReturns(DocstringMeta):
    type_name: Optional[str]
    is_generator: bool
    return_name: Optional[str]

    def __init__(
        self,
        args: List[str],
        description: Optional[str],
        type_name: Optional[str],
        is_generator: bool,
        return_name: Optional[str] = ..., 
    ) -> None: ...


class DocstringYields(DocstringReturns):
    pass


class DocstringRaises(DocstringMeta):
    type_name: Optional[str]

    def __init__(
        self,
        args: List[str],
        description: Optional[str],
        type_name: Optional[str],
    ) -> None: ...


class DocstringDeprecated(DocstringMeta):
    version: Optional[str]

    def __init__(
        self,
        args: List[str],
        description: Optional[str],
        version: Optional[str],
    ) -> None: ...


class DocstringExample(DocstringMeta):
    snippet: Optional[str]

    def __init__(
        self,
        args: List[str],
        snippet: Optional[str],
        description: Optional[str],
    ) -> None: ...


class Docstring:
    short_description: Optional[str]
    long_description: Optional[str]
    blank_after_short_description: bool
    blank_after_long_description: bool
    meta: List[DocstringMeta]
    style: Optional[DocstringStyle]

    def __init__(self, style: Optional[DocstringStyle] = ...) -> None: ...

    @property
    def description(self) -> Optional[str]: ...

    @property
    def params(self) -> List[DocstringParam]: ...

    @property
    def raises(self) -> List[DocstringRaises]: ...

    @property
    def returns(self) -> Optional[DocstringReturns]: ...
