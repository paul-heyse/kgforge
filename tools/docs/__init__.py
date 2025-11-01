"""Curated documentation tooling exports.

The functions and modules re-exported here form the supported interface for generating
project documentation. Failures follow :mod:`kgfoundry_common.errors` and emit Problem
Details payloads consistent with
``schema/examples/tools/problem_details/tool-execution-error.json``.
"""

# ruff: noqa: TID252

from __future__ import annotations

import logging
from importlib import import_module
from typing import Final

from . import (
    build_agent_analytics,
    build_agent_api,
    build_agent_catalog,
    build_graphs,
    build_test_map,
    errors,
    export_schemas,
    observability,
    render_agent_portal,
)

logging.getLogger(__name__).addHandler(logging.NullHandler())

PUBLIC_EXPORTS: Final[dict[str, object]] = {
    "build_agent_analytics": build_agent_analytics,
    "build_agent_api": build_agent_api,
    "build_agent_catalog": build_agent_catalog,
    "build_graphs": build_graphs,
    "build_test_map": build_test_map,
    "errors": errors,
    "export_schemas": export_schemas,
    "observability": observability,
    "render_agent_portal": render_agent_portal,
}

try:
    PUBLIC_EXPORTS["gen_readmes"] = import_module("tools.gen_readmes")
except ImportError as exc:
    message = "Failed to import tools.gen_readmes"
    raise ImportError(message) from exc

__all__: list[str] = sorted(PUBLIC_EXPORTS)
