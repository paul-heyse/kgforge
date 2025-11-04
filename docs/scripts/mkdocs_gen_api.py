"""Public wrapper for :mod:`docs._scripts.mkdocs_gen_api` with lazy loading."""

from __future__ import annotations

from collections.abc import Iterable
from functools import cache
from importlib import import_module
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from types import ModuleType

MODULE_PATH = "docs._scripts.mkdocs_gen_api"

__all__: tuple[str, ...] = ()

_DEFAULT_ALL: tuple[str, ...] = tuple(__all__)


@cache
def _load_module() -> ModuleType:
    return import_module(MODULE_PATH)


@cache
def _export_names() -> tuple[str, ...]:
    module = _load_module()
    exports_obj: object | None = getattr(module, "__all__", None)
    if isinstance(exports_obj, Iterable) and not isinstance(exports_obj, (str, bytes)):
        candidates = cast("Iterable[object]", exports_obj)
        exports: tuple[str, ...] = tuple(str(name) for name in candidates)
    else:
        exports = tuple(name for name in dir(module) if not name.startswith("_"))
    namespace = cast("dict[str, object]", globals())
    namespace["__all__"] = list(exports)
    return exports


def clear_cache() -> None:
    _load_module.cache_clear()
    _export_names.cache_clear()
    namespace = cast("dict[str, object]", globals())
    namespace["__all__"] = list(_DEFAULT_ALL)


def __getattr__(name: str) -> object:
    exports = _export_names()
    if name in exports:
        module = _load_module()
        return cast("object", getattr(module, name))
    message = f"module '{__name__}' has no attribute '{name}'"
    raise AttributeError(message)


def __dir__() -> list[str]:
    return sorted(_export_names())
