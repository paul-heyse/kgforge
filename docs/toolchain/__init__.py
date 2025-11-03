"""Documentation toolchain for KGFoundry.

This package consolidates documentation build scripts and provides a unified
interface for common operations like building symbol indices, generating deltas,
and validating artifacts.
"""

from __future__ import annotations

import logging
from importlib import import_module
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

logging.getLogger(__name__).addHandler(logging.NullHandler())


def build_symbol_index(argv: Sequence[str] | None = None) -> int:
    """Build schema-validated symbol index artifacts for the documentation pipeline.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        Optional CLI-style arguments forwarded to the builder.

    Returns
    -------
    int
        Exit code emitted by the symbol index builder.
    """
    module = import_module("docs._scripts.build_symbol_index")
    main_fn = cast("Callable[[Sequence[str] | None], int]", module.main)
    return main_fn(argv)


def validate_artifacts(argv: Sequence[str] | None = None) -> int:
    """Validate documentation artifacts against defined schemas.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        Optional CLI arguments.

    Returns
    -------
    int
        Exit code from the validation tool.
    """
    module = import_module("docs._scripts.validate_artifacts")
    main_fn = cast("Callable[[Sequence[str] | None], int]", module.main)
    return main_fn(argv)


def symbol_delta(argv: Sequence[str] | None = None) -> int:
    """Generate symbol index delta between two builds.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        Optional CLI arguments.

    Returns
    -------
    int
        Exit code from the delta generator.
    """
    module = import_module("docs._scripts.symbol_delta")
    main_fn = cast("Callable[[Sequence[str] | None], int]", module.main)
    return main_fn(argv)


_PUBLIC_EXPORTS: dict[str, object] = {
    "build_symbol_index": build_symbol_index,
    "symbol_delta": symbol_delta,
    "validate_artifacts": validate_artifacts,
}

PUBLIC_EXPORTS: Final[Mapping[str, object]] = MappingProxyType(_PUBLIC_EXPORTS)

__all__: tuple[str, ...] = (
    "build_symbol_index",
    "symbol_delta",
    "validate_artifacts",
)
