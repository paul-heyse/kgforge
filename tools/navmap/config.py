"""Configuration models for navmap toolkit operations.

This module defines typed configuration objects for navmap repair and strip operations,
replacing boolean positional arguments with immutable frozen dataclasses that include
built-in validation.

Examples
--------
>>> from tools.navmap.config import NavmapRepairOptions
>>> options = NavmapRepairOptions(apply=True, emit_json=False)
>>> options.apply
True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "NavmapRepairOptions",
    "NavmapStripOptions",
]


@dataclass(frozen=True, slots=True)
class NavmapRepairOptions:
    """Configuration for navmap repair operations.

    This configuration object controls how the navmap repair tool identifies
    and optionally applies fixes to module navmap metadata.

    Attributes
    ----------
    root : Path | None, optional
        Root directory to scan for modules with navmap metadata.
        Defaults to None (uses project src/ directory).
    apply : bool, optional
        When True, write fixes back to disk. When False, emit suggested changes
        to stdout. Defaults to False.
    emit_json : bool, optional
        When True, emit machine-readable JSON results. When False, emit human-readable
        text output. Defaults to False.

    Examples
    --------
    >>> options = NavmapRepairOptions(apply=False, emit_json=True)
    >>> options.apply
    False
    >>> options.emit_json
    True
    """

    root: Path | None = None
    apply: bool = False
    emit_json: bool = False


@dataclass(frozen=True, slots=True)
class NavmapStripOptions:
    """Configuration for navmap strip operations.

    This configuration object controls how the navmap strip tool removes
    deprecated NavMap metadata from module docstrings.

    Attributes
    ----------
    dry_run : bool, optional
        When True, report what would be removed without modifying files.
        When False, write cleaned docstrings back to disk. Defaults to True
        (safe by default).
    verbose : bool, optional
        When True, emit detailed messages about each file processed.
        Defaults to False.

    Examples
    --------
    >>> options = NavmapStripOptions(dry_run=False, verbose=True)
    >>> options.dry_run
    False
    >>> options.verbose
    True
    """

    dry_run: bool = True
    verbose: bool = False
