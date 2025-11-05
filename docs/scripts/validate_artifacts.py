"""Public wrapper for :mod:`docs._scripts.validate_artifacts` with lazy loading."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, cast

from docs.scripts._module_cache import ModuleExportsCache

if TYPE_CHECKING:
    from types import ModuleType
else:  # pragma: no cover - runtime helper for annotations
    ModuleType = type(import_module("types"))

MODULE_PATH = "docs._scripts.validate_artifacts"

__all__: tuple[str, ...] = ()

_DEFAULT_ALL: tuple[str, ...] = tuple(__all__)


_CACHE = ModuleExportsCache(MODULE_PATH)


def _load_module() -> ModuleType:
    return _CACHE.load_module()


def _export_names() -> tuple[str, ...]:
    exports = _CACHE.export_names()
    namespace = cast("dict[str, object]", globals())
    namespace["__all__"] = list(exports)
    return exports


def clear_cache() -> None:
    """Clear the module cache and reset exports.

    Resets the cached module and restores the default __all__ list.
    """
    _CACHE.reset()
    namespace = cast("dict[str, object]", globals())
    namespace["__all__"] = list(_DEFAULT_ALL)


def __getattr__(name: str) -> object:
    """Get module attribute via lazy import.

    Parameters
    ----------
    name : str
        Attribute name to import.

    Returns
    -------
    object
        Imported attribute.

    Raises
    ------
    AttributeError
        If the attribute name is not in exports.
    """
    exports = _export_names()
    if name in exports:
        module = _load_module()
        return cast("object", getattr(module, name))
    message = f"module '{__name__}' has no attribute '{name}'"
    raise AttributeError(message)


def __dir__() -> list[str]:
    """Return list of available module attributes.

    Returns
    -------
    list[str]
        Sorted list of exported attribute names.
    """
    return sorted(_export_names())
