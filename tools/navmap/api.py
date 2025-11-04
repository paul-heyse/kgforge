"""Public API for navmap repair operations with typed configuration.

This module provides the new typed configuration-based API for navmap repairs,
wrapping existing functions while maintaining backward compatibility.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Protocol, cast

from tools.navmap import repair_navmaps
from tools.navmap.repair_navmaps import RepairExecutionConfig

if TYPE_CHECKING:
    from pathlib import Path

    from tools.navmap.build_navmap import ModuleInfo
    from tools.navmap.config import NavmapRepairOptions


class ModuleInfoProtocol(Protocol):
    """Protocol interface for ModuleInfo at runtime."""

    path: Path
    name: str


class NavmapRepairOptionsProtocol(Protocol):
    """Protocol interface for NavmapRepairOptions at runtime."""

    root: Path | None
    apply: bool
    emit_json: bool


class RepairResultProtocol(Protocol):
    """Protocol interface for RepairResult at runtime."""

    module: Path
    messages: list[str]
    changed: bool
    applied: bool


class RepairExecutionConfigProtocol(Protocol):
    """Protocol interface for RepairExecutionConfig at runtime."""

    apply_changes: bool
    emit_json: bool


def repair_module_with_config(
    info: ModuleInfoProtocol,
    *,
    options: NavmapRepairOptionsProtocol,
) -> RepairResultProtocol:
    """Repair a module's navmap metadata with typed configuration.

    This is the new public API for module repair that accepts a typed
    configuration object instead of boolean positional arguments.

    Parameters
    ----------
    info : ModuleInfoProtocol
        Metadata describing the target module discovered by ``build_navmap``.
    options : NavmapRepairOptionsProtocol
        Typed configuration controlling repair behavior.

    Returns
    -------
    RepairResultProtocol
        Outcome describing emitted messages alongside change and apply flags.

    Examples
    --------
    >>> from tools.navmap.config import NavmapRepairOptions
    >>> options = NavmapRepairOptions(apply=True)
    >>> # result = repair_module_with_config(module_info, options=options)
    """
    typed_options = cast("NavmapRepairOptions", options)
    execution = RepairExecutionConfig(
        apply_changes=typed_options.apply,
        emit_json=typed_options.emit_json,
    )
    typed_info = cast("ModuleInfo", info)
    result = repair_navmaps.repair_module(typed_info, execution=execution)
    return cast("RepairResultProtocol", result)


def repair_all_with_config(
    *,
    root: Path | None = None,
    options: NavmapRepairOptionsProtocol,
) -> list[RepairResultProtocol]:
    """Repair every module under a root directory with typed configuration.

    This is the new public API for batch repair that accepts a typed
    configuration object instead of boolean positional arguments.

    Parameters
    ----------
    root : Path | None, optional
        Directory tree to scan. Defaults to project src/ if None.
    options : NavmapRepairOptionsProtocol
        Typed configuration controlling repair behavior (apply, etc.).

    Returns
    -------
    list[RepairResultProtocol]
        List of repair results for all modules found under root.

    Examples
    --------
    >>> from pathlib import Path
    >>> from tools.navmap.config import NavmapRepairOptions
    >>> options = NavmapRepairOptions(apply=False)
    >>> # results = repair_all_with_config(root=Path("src"), options=options)
    """
    typed_options = cast("NavmapRepairOptions", options)
    target_root = root or typed_options.root or repair_navmaps.SRC
    execution = RepairExecutionConfig(
        apply_changes=typed_options.apply,
        emit_json=typed_options.emit_json,
    )
    results = repair_navmaps.repair_all(target_root, execution=execution)
    return cast("list[RepairResultProtocol]", results)


def repair_all_legacy(*, root: Path, apply: bool) -> list[RepairResultProtocol]:
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
    list[RepairResultProtocol]
        List of repair results for all modules found.
    """
    msg = (
        "repair_all_legacy() is deprecated. "
        "Use repair_all_with_config(root=..., options=NavmapRepairOptions(...)) instead."
    )
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    execution = RepairExecutionConfig(apply_changes=apply, emit_json=False)
    results = repair_navmaps.repair_all(root, execution=execution)
    return cast("list[RepairResultProtocol]", results)


__all__ = [
    "repair_all_legacy",
    "repair_all_with_config",
    "repair_module_with_config",
]
