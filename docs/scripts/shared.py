"""Public wrapper for :mod:`docs._scripts.shared` with lazy loading."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, cast

from docs.scripts._module_cache import ModuleExportsCache

MODULE_PATH = "docs._scripts.shared"

__all__ = [
    "BuildEnvironment",
    "DocsSettings",
    "GriffeLoader",
    "LinkMode",
    "WarningLogger",
    "build_warning_logger",
    "detect_environment",
    "ensure_sys_paths",
    "load_settings",
    "make_loader",
    "make_logger",
    "resolve_git_sha",
    "safe_json_deserialize",
    "safe_json_serialize",
]

_CACHE = ModuleExportsCache(MODULE_PATH)


def _load_module() -> ModuleType:
    return _CACHE.load_module()


def clear_cache() -> None:
    _CACHE.reset()


def __getattr__(name: str) -> object:
    if name in __all__:
        module = _load_module()
        return cast("object", getattr(module, name))
    message = f"module '{__name__}' has no attribute '{name}'"
    raise AttributeError(message)


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:
    from types import ModuleType
else:  # pragma: no cover - runtime helper for annotations
    ModuleType = type(import_module("types"))

BuildEnvironment: object
DocsSettings: object
GriffeLoader: object
LinkMode: object
WarningLogger: object
build_warning_logger: object
detect_environment: object
ensure_sys_paths: object
load_settings: object
make_loader: object
make_logger: object
resolve_git_sha: object
safe_json_deserialize: object
safe_json_serialize: object
