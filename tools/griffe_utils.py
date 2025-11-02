"""Shared helpers for working with ``griffe`` across docs tooling.

These utilities allow us to tolerate changes in the ``griffe`` package layout.
Newer releases ship ``GriffeLoader`` and the core symbol dataclasses from the
top-level package rather than the ``griffe.loader`` and ``griffe.dataclasses``
modules. We normalise the import logic in one place so scripts, tests, and the
Sphinx configuration can rely on a consistent interface.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any, cast

GriffeModuleType = ModuleType
GriffeObjectType = type[Any]
GriffeLoaderType = type[Any]
GriffeClassType = type[Any]
GriffeFunctionType = type[Any]
GriffeModuleObjectType = type[Any]


@dataclass(frozen=True, slots=True)
class GriffeAPI:
    """Container exposing the bits of ``griffe`` required by tooling."""

    package: GriffeModuleType
    object_type: GriffeObjectType
    loader_type: GriffeLoaderType
    class_type: GriffeClassType
    function_type: GriffeFunctionType
    module_type: GriffeModuleObjectType


_GRIFFE_SINGLETON: GriffeAPI | None = None


def resolve_griffe() -> GriffeAPI:
    """Return the ``griffe`` module along with key symbol/loader types.

    Newer versions of ``griffe`` expose ``GriffeLoader`` and the symbol classes
    (``Class``, ``Function``, and ``Module``) from the top-level package while
    older releases provide them under ``griffe.loader`` and
    ``griffe.dataclasses`` respectively. We import both variants eagerly and
    fall back to the top-level attributes whenever the nested modules are
    unavailable.
    """
    global _GRIFFE_SINGLETON

    if _GRIFFE_SINGLETON is not None:
        return _GRIFFE_SINGLETON

    griffe_module = importlib.import_module("griffe")
    try:
        loader_module = importlib.import_module("griffe.loader")
    except ModuleNotFoundError:
        loader_module = None
    try:
        dataclasses_module = importlib.import_module("griffe.dataclasses")
    except ModuleNotFoundError:
        dataclasses_module = None

    if loader_module is not None and hasattr(loader_module, "GriffeLoader"):
        loader_attr: object = loader_module.GriffeLoader
    else:
        loader_attr = griffe_module.GriffeLoader
    loader_cls = cast(GriffeLoaderType, loader_attr)

    symbols_source: ModuleType = (
        dataclasses_module if dataclasses_module is not None else griffe_module
    )
    class_attr: object = symbols_source.Class
    function_attr: object = symbols_source.Function
    module_attr: object = symbols_source.Module
    object_attr: object = griffe_module.Object

    class_cls = cast(GriffeClassType, class_attr)
    function_cls = cast(GriffeFunctionType, function_attr)
    module_cls = cast(GriffeModuleObjectType, module_attr)
    object_cls = cast(GriffeObjectType, object_attr)

    _GRIFFE_SINGLETON = GriffeAPI(
        package=griffe_module,
        object_type=object_cls,
        loader_type=loader_cls,
        class_type=class_cls,
        function_type=function_cls,
        module_type=module_cls,
    )
    return _GRIFFE_SINGLETON
