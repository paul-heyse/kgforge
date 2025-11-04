"""Public wrapper for :mod:`docs._scripts.mkdocs_gen_api` with lazy loading."""

from __future__ import annotations

from collections.abc import Iterable
from importlib import import_module
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from types import ModuleType

MODULE_PATH = "docs._scripts.mkdocs_gen_api"

__all__: tuple[str, ...] = ()


_MODULE_CACHE: ModuleType | None = None
_EXPORTS_CACHE: tuple[str, ...] | None = None


def _load_module() -> ModuleType:
    global _MODULE_CACHE
    if _MODULE_CACHE is None:
        _MODULE_CACHE = import_module(MODULE_PATH)
    return _MODULE_CACHE


def _export_names() -> tuple[str, ...]:
    global _EXPORTS_CACHE
    if _EXPORTS_CACHE is not None:
        return _EXPORTS_CACHE

    module = _load_module()
    exports_obj: object | None = getattr(module, "__all__", None)
    if isinstance(exports_obj, Iterable) and not isinstance(exports_obj, (str, bytes)):
        exports: tuple[str, ...] = tuple(str(name) for name in exports_obj)
    else:
        exports = tuple(name for name in dir(module) if not name.startswith("_"))
    namespace = cast("dict[str, object]", globals())
    namespace["__all__"] = list(exports)
    _EXPORTS_CACHE = exports
    return exports


def __getattr__(name: str) -> object:
    exports = _export_names()
    if name in exports:
        module = _load_module()
        return cast("object", getattr(module, name))
    message = f"module '{__name__}' has no attribute '{name}'"
    raise AttributeError(message)


def __dir__() -> list[str]:
    return sorted(_export_names())
