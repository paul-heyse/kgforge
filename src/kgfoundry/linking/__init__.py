"""Namespace bridge that exposes the linking package under kgfoundry."""

from __future__ import annotations

import linking as _module
from kgfoundry._namespace_proxy import namespace_dir, namespace_getattr

__all__ = list(getattr(_module, "__all__", []))
if not __all__:
    __all__ = [name for name in dir(_module) if not name.startswith("_")]
for _name in __all__:
    globals()[_name] = getattr(_module, _name)

__doc__ = _module.__doc__
if hasattr(_module, "__path__"):
    __path__ = list(_module.__path__)


def __getattr__(name: str) -> object:
    """Document   getattr  .

    <!-- auto:docstring-builder v1 -->

    Provide a fallback for unknown attribute lookups. This special method integrates the class with Python's data model so instances behave consistently with the language expectations.

    Parameters
    ----------
    name : str
        Describe `name`.


    Returns
    -------
    object
        Describe return value.
    """
    return namespace_getattr(_module, name)


def __dir__() -> list[str]:
    """Return the combined attribute listing.

    <!-- auto:docstring-builder v1 -->

    Returns
    -------
    inspect._empty
        Sorted union of exports and implementation attributes.
    """
    return namespace_dir(_module, __all__)
