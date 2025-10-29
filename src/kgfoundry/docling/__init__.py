"""Namespace bridge that exposes the docling package under kgfoundry."""

from __future__ import annotations

from typing import Any

import docling as _module

__all__ = list(getattr(_module, "__all__", []))
if not __all__:
    __all__ = [name for name in dir(_module) if not name.startswith("_")]
for _name in __all__:
    globals()[_name] = getattr(_module, _name)

__doc__ = _module.__doc__
if hasattr(_module, "__path__"):
    __path__ = list(_module.__path__)


def __getattr__(name: str) -> Any:
    """Document   getattr  .

<!-- auto:docstring-builder v1 -->

Provide a fallback for unknown attribute lookups. This special method integrates the class with Python's data model so instances behave consistently with the language expectations.

Parameters
----------
name : str
    TODO: describe ``name``.


Returns
-------
Any
    TODO: describe return value.
"""
    return getattr(_module, name)


def __dir__() -> list[str]:
    """Document   dir  .

<!-- auto:docstring-builder v1 -->

Expose the attributes reported when ``dir()`` is called on the instance. This special method integrates the class with Python's data model so instances behave consistently with the language expectations.


Returns
-------
inspect._empty
    Describe return value.
"""
    candidates = set(__all__)
    candidates.update(name for name in dir(_module) if not name.startswith("__"))
    return sorted(candidates)
