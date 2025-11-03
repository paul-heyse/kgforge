"""Public API for navmap repair operations with typed configuration.

This module provides the new typed configuration-based API for navmap repairs,
wrapping existing functions while maintaining backward compatibility.
"""

from __future__ import annotations

import warnings
from pathlib import Path  # noqa: TC003

from tools.navmap import repair_navmaps
from tools.navmap.config import NavmapRepairOptions  # noqa: TC001
from tools.navmap.repair_navmaps import ModuleInfo, RepairResult  # noqa: TC001

__all__ = [
    "repair_all_legacy",
    "repair_all_with_config",
    "repair_module_with_config",
]


def repair_module_with_config(
    info: ModuleInfo,
    *,
    options: NavmapRepairOptions,
) -> RepairResult:
    """Repair a module's navmap metadata with typed configuration.

    This is the new public API for module repair that accepts a typed
    configuration object instead of boolean positional arguments.

    Parameters
    ----------
    info : ModuleInfo
        Metadata describing the target module discovered by ``build_navmap``.
    options : NavmapRepairOptions
        Typed configuration controlling repair behavior.

    Returns
    -------
    RepairResult
        Outcome describing emitted messages alongside change and apply flags.

    Examples
    --------
    >>> from tools.navmap.config import NavmapRepairOptions
    >>> options = NavmapRepairOptions(apply=True)
    >>> # result = repair_module_with_config(module_info, options=options)
    """
    return repair_navmaps.repair_module(info, apply=options.apply)


def repair_all_with_config(
    *,
    root: Path | None = None,
    options: NavmapRepairOptions,
) -> list[RepairResult]:
    """Repair every module under a root directory with typed configuration.

    This is the new public API for batch repair that accepts a typed
    configuration object instead of boolean positional arguments.

    Parameters
    ----------
    root : Path | None, optional
        Directory tree to scan. Defaults to project src/ if None.
    options : NavmapRepairOptions
        Typed configuration controlling repair behavior (apply, etc.).

    Returns
    -------
    list[RepairResult]
        List of repair results for all modules found under root.

    Examples
    --------
    >>> from pathlib import Path
    >>> from tools.navmap.config import NavmapRepairOptions
    >>> options = NavmapRepairOptions(apply=False)
    >>> # results = repair_all_with_config(root=Path("src"), options=options)
    """
    target_root = root or repair_navmaps.SRC
    return repair_navmaps.repair_all(target_root, apply=options.apply)


def repair_all_legacy(*, root: Path, apply: bool) -> list[RepairResult]:
    """Deprecate legacy repair_all API in favor of typed configuration.

    .. deprecated::
        Use :func:`repair_all_with_config` with typed configuration objects.

    This function is provided for backward compatibility and will be removed
    in a future release.

    Parameters
    ----------
    root : Path
        Directory tree to scan for modules with navmap metadata.
    apply : bool
        When True, write fixes to disk. When False, report changes.

    Returns
    -------
    list[RepairResult]
        List of repair results for all modules found.
    """
    msg = (
        "repair_all_legacy() is deprecated. "
        "Use repair_all_with_config(root=..., options=NavmapRepairOptions(...)) instead."
    )
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    return repair_navmaps.repair_all(root, apply=apply)
