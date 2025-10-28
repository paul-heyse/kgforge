"""Shared helpers for working with ``griffe`` across docs tooling.

These utilities allow us to tolerate changes in the ``griffe`` package layout.
Newer releases ship ``GriffeLoader`` from the top-level package rather than the
``griffe.loader`` module. We normalise the import logic in one place so scripts,
tests, and the Sphinx configuration can rely on a consistent interface.
"""

from __future__ import annotations

import importlib
from functools import cache
from types import ModuleType
from typing import Any, TypeAlias

GriffeModuleType: TypeAlias = ModuleType
GriffeObjectType: TypeAlias = type[Any]
GriffeLoaderType: TypeAlias = type[Any]


@cache
def resolve_griffe() -> tuple[GriffeModuleType, GriffeObjectType, GriffeLoaderType]:
    """Return the ``griffe`` module, ``Object`` type, and ``GriffeLoader`` class.

    Newer versions of ``griffe`` expose ``GriffeLoader`` from the top-level
    package while older versions keep it under ``griffe.loader``. We import both
    variants eagerly and fall back to the top-level attribute whenever the
    nested module is unavailable.
    """
    griffe_module = importlib.import_module("griffe")
    try:
        loader_module = importlib.import_module("griffe.loader")
    except ModuleNotFoundError:
        loader_module = None

    loader_cls = getattr(loader_module, "GriffeLoader", getattr(griffe_module, "GriffeLoader"))
    object_cls = getattr(griffe_module, "Object")
    return griffe_module, object_cls, loader_cls
