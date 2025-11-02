"""Typing surface for the docstring builder package."""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType

cache: ModuleType
cli: ModuleType
config: ModuleType
docfacts: ModuleType
harvest: ModuleType
ir: ModuleType
models: ModuleType
observability: ModuleType
policy: ModuleType
render: ModuleType
schema: ModuleType
semantics: ModuleType

BUILDER_VERSION: str

CliMain = Callable[[list[str] | None], int]

def main(argv: list[str] | None = ...) -> int: ...

__all__: tuple[str, ...]
