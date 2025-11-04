"""Consolidated CLI tooling for KGFoundry.

This package provides a unified interface for command-line tools including documentation builders,
navigation map utilities, and code quality checkers.
"""

from __future__ import annotations

import logging
from importlib import import_module
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

logging.getLogger(__name__).addHandler(logging.NullHandler())


def build_agent_catalog(argv: Sequence[str] | None = None) -> int:
    """Construct the agent catalog JSON artifact.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        Optional CLI arguments.

    Returns
    -------
    int
        Exit code from the catalog builder.
    """
    module = import_module("tools.docs.build_agent_catalog")
    main_fn = cast("Callable[[Sequence[str] | None], int]", module.main)
    return main_fn(argv)


def build_navmap(argv: Sequence[str] | None = None) -> int:
    """Build navigation map artifacts for the documentation site.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        Optional CLI arguments.

    Returns
    -------
    int
        Exit code from the navmap builder.
    """
    module = import_module("tools.navmap.build_navmap")
    main_fn = cast("Callable[[Sequence[str] | None], int]", module.main)
    return main_fn(argv)


_PUBLIC_EXPORTS: dict[str, object] = {
    "build_agent_catalog": build_agent_catalog,
    "build_navmap": build_navmap,
}

PUBLIC_EXPORTS: Final[Mapping[str, object]] = MappingProxyType(_PUBLIC_EXPORTS)

__all__: tuple[str, ...] = (
    "build_agent_catalog",
    "build_navmap",
)
