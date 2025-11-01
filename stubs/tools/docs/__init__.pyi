"""Typing facade for documentation tooling exports."""

from __future__ import annotations

from types import ModuleType
from typing import Final

build_agent_analytics: ModuleType
build_agent_api: ModuleType
build_agent_catalog: ModuleType
build_graphs: ModuleType
build_test_map: ModuleType
errors: ModuleType
export_schemas: ModuleType
gen_readmes: ModuleType
observability: ModuleType
render_agent_portal: ModuleType

PUBLIC_EXPORTS: Final[dict[str, object]]

__all__ = sorted(
    [
        "build_agent_analytics",
        "build_agent_api",
        "build_agent_catalog",
        "build_graphs",
        "build_test_map",
        "errors",
        "export_schemas",
        "gen_readmes",
        "observability",
        "render_agent_portal",
    ]
)
