"""Public wrapper for :mod:`docs._scripts.shared` with lazy loading."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from types import ModuleType

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


def _load_module() -> ModuleType:
    module = cast("ModuleType | None", getattr(_load_module, "_cache", None))
    if module is None:
        module = import_module(MODULE_PATH)
        _load_module._cache = module  # type: ignore[attr-defined] # noqa: SLF001
    return module


def __getattr__(name: str) -> object:
    if name in __all__:
        module = _load_module()
        return cast("object", getattr(module, name))
    message = f"module '{__name__}' has no attribute '{name}'"
    raise AttributeError(message)


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # pragma: no cover - typing assistance only
    from docs._scripts import shared as _shared

    BuildEnvironment = _shared.BuildEnvironment
    DocsSettings = _shared.DocsSettings
    GriffeLoader = _shared.GriffeLoader
    LinkMode = _shared.LinkMode
    WarningLogger = _shared.WarningLogger
    build_warning_logger = _shared.build_warning_logger
    detect_environment = _shared.detect_environment
    ensure_sys_paths = _shared.ensure_sys_paths
    load_settings = _shared.load_settings
    make_loader = _shared.make_loader
    make_logger = _shared.make_logger
    resolve_git_sha = _shared.resolve_git_sha
    safe_json_deserialize = _shared.safe_json_deserialize
    safe_json_serialize = _shared.safe_json_serialize
