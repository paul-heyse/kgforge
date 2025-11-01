"""Typing facade for documentation tooling exports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import ModuleType
from typing import Final

def build_agent_analytics(argv: Sequence[str] | None = ...) -> int: ...
def build_agent_api() -> int: ...
def build_agent_catalog(argv: Sequence[str] | None = ...) -> int: ...
def build_artifacts() -> int: ...
def build_graphs() -> None: ...
def build_test_map() -> None: ...
def export_schemas(argv: Sequence[str] | None = ...) -> int: ...
def render_agent_portal(argv: Sequence[str] | None = ...) -> int: ...
def scan_observability() -> int: ...

catalog_models: ModuleType
errors: ModuleType
observability: ModuleType

PUBLIC_EXPORTS: Final[Mapping[str, object]]
MODULE_EXPORTS: Final[Mapping[str, str]]

__all__ = sorted(
    [
        "build_agent_analytics",
        "build_agent_api",
        "build_agent_catalog",
        "build_artifacts",
        "build_graphs",
        "build_test_map",
        "catalog_models",
        "errors",
        "export_schemas",
        "observability",
        "render_agent_portal",
        "scan_observability",
    ]
)
