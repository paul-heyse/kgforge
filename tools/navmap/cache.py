"""Public cache interface for navmap operations.

This module provides cache interfaces and implementations for tracking module
metadata and repair results during navmap operations.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from tools.navmap.build_navmap import ModuleInfo
    from tools.navmap.repair_navmaps import RepairResult


@runtime_checkable
class NavmapCollectorCache(Protocol):
    """Public cache interface for module metadata collection operations.

    This Protocol defines the contract for any object that implements
    module collection caching. Implementations must track module metadata
    and provide access to collected modules.

    Attributes
    ----------
    root : Path
        The root directory being scanned for modules.
    """

    @property
    def root(self) -> Path:
        """Root directory being scanned.

        Returns
        -------
        Path
            The root directory from which modules are collected.
        """
        ...

    def collect_modules(self) -> list[ModuleInfo]:
        """Collect all modules under the root directory.

        Discovers all Python modules under the root directory and returns
        their metadata. Results may be cached between calls with the same
        root directory if no changes are detected.

        Returns
        -------
        list[ModuleInfo]
            List of module metadata objects discovered under root.

        Examples
        --------
        >>> # cache.collect_modules()
        [ModuleInfo(...), ModuleInfo(...)]
        """
        ...

    def get_module(self, path: Path) -> ModuleInfo | None:
        """Retrieve metadata for a specific module.

        Parameters
        ----------
        path : Path
            Path to the module file.

        Returns
        -------
        ModuleInfo | None
            Module metadata if found, None otherwise.

        Examples
        --------
        >>> # cache.get_module(Path("src/example.py"))
        ModuleInfo(...)
        """
        ...


@runtime_checkable
class NavmapRepairCache(Protocol):
    """Public cache interface for module repair operation tracking.

    This Protocol defines the contract for objects that track the results
    of repair operations on modules.
    """

    def record_repair(self, result: RepairResult) -> None:
        """Record the result of a repair operation.

        Parameters
        ----------
        result : RepairResult
            The result of repairing a single module.

        Examples
        --------
        >>> # cache.record_repair(repair_result)
        """
        ...

    def get_repairs(self) -> list[RepairResult]:
        """Retrieve all recorded repair results.

        Returns
        -------
        list[RepairResult]
            List of repair results recorded so far.

        Examples
        --------
        >>> # cache.get_repairs()
        [RepairResult(...), RepairResult(...)]
        """
        ...

    def summary(self) -> dict[str, int]:
        """Get a summary of repair statistics.

        Returns
        -------
        dict[str, int]
            Dictionary with statistics (e.g., total, changed, applied counts).

        Examples
        --------
        >>> # cache.summary()
        {"total": 10, "changed": 3, "applied": 2}
        """
        ...


__all__ = [
    "NavmapCollectorCache",
    "NavmapRepairCache",
]
