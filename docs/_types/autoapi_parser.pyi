from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Protocol

__all__ = ["AutoapiParserProtocol", "coerce_parser_class"]

class AutoapiParserProtocol(Protocol):
    def parse(self, node: object, /) -> object: ...
    def _parse_file(self, file_path: str, condition: Callable[[str], bool], /) -> object: ...
    def parse_file(self, file_path: str, /) -> object: ...
    def parse_file_in_namespace(self, file_path: str, dir_root: str, /) -> object: ...

def coerce_parser_class(
    module: ModuleType,
    attribute: str = ...,
) -> type[AutoapiParserProtocol]: ...
