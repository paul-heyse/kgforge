"""Curated documentation tooling exports.

Install the optional extra ``kgfoundry[tools]`` when consuming the project as a wheel
to access these helpers. Failures surface as Problem Details consistent with
``schema/examples/tools/problem_details/tool-execution-error.json``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from importlib import import_module
from types import MappingProxyType, ModuleType
from typing import TYPE_CHECKING, Final, cast

logging.getLogger(__name__).addHandler(logging.NullHandler())


def _coerce_argv(argv: Sequence[str] | None) -> list[str] | None:
    """Return a list copy of ``argv`` or ``None`` when no arguments are provided."""
    return list(argv) if argv is not None else None


def build_agent_analytics(argv: Sequence[str] | None = None) -> int:
    """Build the agent analytics documentation artifacts.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        Optional CLI-style arguments forwarded to the underlying builder.

    Returns
    -------
    int
        Exit code emitted by the analytics builder.

    Notes
    -----
    Errors propagate as structured Problem Details documented under
    ``schema/examples/tools/problem_details/tool-execution-error.json``.
    """
    module = import_module("tools.docs.build_agent_analytics")
    main_fn = cast(Callable[[list[str] | None], int], module.main)
    return main_fn(_coerce_argv(argv))


def build_agent_api() -> int:
    """Generate the Agent API documentation bundle.

    Returns
    -------
    int
        Exit code emitted by the Agent API builder.

    Notes
    -----
    Failures emit Problem Details aligned with
    ``schema/examples/tools/problem_details/tool-execution-error.json``.
    """
    module = import_module("tools.docs.build_agent_api")
    main_fn = cast(Callable[[], int], module.main)
    return main_fn()


def build_agent_catalog(argv: Sequence[str] | None = None) -> int:
    """Construct the agent catalog JSON artifact consumed by downstream docs.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        Optional CLI arguments forwarded to the catalog builder.

    Returns
    -------
    int
        Exit code reported by the catalog builder.

    Notes
    -----
    Exceptions are rendered as Problem Details compatible with
    ``schema/examples/tools/problem_details/tool-execution-error.json``.
    """
    module = import_module("tools.docs.build_agent_catalog")
    main_fn = cast(Callable[[list[str] | None], int], module.main)
    return main_fn(_coerce_argv(argv))


def build_artifacts() -> int:
    """Assemble the full documentation artifact set.

    Returns
    -------
    int
        Exit code returned by the documentation artifact builder.

    Notes
    -----
    Structured failures follow the Problem Details exemplar stored under
    ``schema/examples/tools/problem_details/tool-execution-error.json``.
    """
    module = import_module("tools.docs.build_artifacts")
    main_fn = cast(Callable[[], int], module.main)
    return main_fn()


def build_graphs() -> None:
    """Render dependency graphs for the project documentation suite.

    Notes
    -----
    Errors are surfaced as Problem Details consistent with
    ``schema/examples/tools/problem_details/tool-execution-error.json``.
    """
    module = import_module("tools.docs.build_graphs")
    main_fn = cast(Callable[[], None], module.main)
    main_fn()


def build_test_map() -> None:
    """Generate the documentation test mapping artifact.

    Notes
    -----
    Structured failures align with
    ``schema/examples/tools/problem_details/tool-execution-error.json``.
    """
    module = import_module("tools.docs.build_test_map")
    main_fn = cast(Callable[[], None], module.main)
    main_fn()


def export_schemas(argv: Sequence[str] | None = None) -> int:
    """Export JSON Schemas used by documentation tooling.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        Optional CLI-style arguments passed through to the schema exporter.

    Returns
    -------
    int
        Exit code returned by the schema exporter.

    Notes
    -----
    Structured errors align with
    ``schema/examples/tools/problem_details/tool-execution-error.json``.
    """
    module = import_module("tools.docs.export_schemas")
    main_fn = cast(Callable[[list[str] | None], int], module.main)
    return main_fn(_coerce_argv(argv))


def render_agent_portal(argv: Sequence[str] | None = None) -> int:
    """Render the Agent Portal static site.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        Optional CLI-style arguments forwarded to the portal renderer.

    Returns
    -------
    int
        Exit code returned by the portal renderer.

    Notes
    -----
    Failures emit Problem Details aligned with
    ``schema/examples/tools/problem_details/tool-execution-error.json``.
    """
    module = import_module("tools.docs.render_agent_portal")
    main_fn = cast(Callable[[list[str] | None], int], module.main)
    return main_fn(_coerce_argv(argv))


def scan_observability() -> int:
    """Scan observability metadata and emit compliance reports.

    Returns
    -------
    int
        Exit code emitted by the observability scanner.

    Notes
    -----
    Exceptions are serialised as Problem Details consistent with
    ``schema/examples/tools/problem_details/tool-execution-error.json``.
    """
    module = import_module("tools.docs.scan_observability")
    main_fn = cast(Callable[[], int], module.main)
    return main_fn()


_PUBLIC_EXPORTS: dict[str, object] = {
    "build_agent_analytics": build_agent_analytics,
    "build_agent_api": build_agent_api,
    "build_agent_catalog": build_agent_catalog,
    "build_artifacts": build_artifacts,
    "build_graphs": build_graphs,
    "build_test_map": build_test_map,
    "export_schemas": export_schemas,
    "render_agent_portal": render_agent_portal,
    "scan_observability": scan_observability,
}

PUBLIC_EXPORTS: Final[Mapping[str, object]] = MappingProxyType(_PUBLIC_EXPORTS)

_MODULE_EXPORTS: dict[str, str] = {
    "catalog_models": "tools.docs.catalog_models",
    "errors": "tools.docs.errors",
    "observability": "tools.docs.observability",
}

MODULE_EXPORTS: Final[Mapping[str, str]] = MappingProxyType(_MODULE_EXPORTS)

if TYPE_CHECKING:
    from tools.docs import catalog_models, errors, observability

__all__: tuple[str, ...] = (
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
)


def __getattr__(name: str) -> ModuleType:
    if name in MODULE_EXPORTS:
        module = import_module(MODULE_EXPORTS[name])
        module_globals = cast(dict[str, object], globals())
        module_globals[name] = module
        return module
    message = f"module 'tools.docs' has no attribute {name!r}"
    raise AttributeError(message)


def __dir__() -> list[str]:
    module_globals = cast(dict[str, object], globals())
    return sorted({*module_globals, *MODULE_EXPORTS.keys()})
