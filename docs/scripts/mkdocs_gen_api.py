"""Public wrapper for :mod:`docs._scripts.mkdocs_gen_api` with lazy loading."""

from __future__ import annotations

from collections.abc import Iterable
from importlib import import_module
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from types import ModuleType

MODULE_PATH = "docs._scripts.mkdocs_gen_api"

__all__: tuple[str, ...] = ()


def _load_module() -> ModuleType:
    module = cast("ModuleType | None", getattr(_load_module, "_cache", None))
    if module is None:
        module = import_module(MODULE_PATH)
        _load_module._cache = module  # type: ignore[attr-defined] # noqa: SLF001
    return module


def _export_names() -> tuple[str, ...]:
    exports = cast("tuple[str, ...] | None", getattr(_export_names, "_cache", None))
    if exports is None:
        module = _load_module()
        exports_obj: object | None = getattr(module, "__all__", None)
        if isinstance(exports_obj, Iterable) and not isinstance(exports_obj, (str, bytes)):
            names = tuple(str(name) for name in exports_obj)
        else:
            names = tuple(name for name in dir(module) if not name.startswith("_"))
        exports = names
        _export_names._cache = exports  # type: ignore[attr-defined] # noqa: SLF001
        namespace = cast("dict[str, object]", globals())
        namespace["__all__"] = list(exports)
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
