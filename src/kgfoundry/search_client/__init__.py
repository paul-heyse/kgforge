"""Namespace bridge that exposes the ``search_client`` package under
``kgfoundry``.
"""

from __future__ import annotations

from typing import Any

import search_client as _module
from search_client import KGFoundryClient as _KGFoundryClient

KGFoundryClient = _KGFoundryClient

__all__ = ["KGFoundryClient"]
__doc__ = _module.__doc__
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
