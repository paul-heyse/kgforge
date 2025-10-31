"""Namespace bridge that exposes the ontology package under kgfoundry."""

from __future__ import annotations

import ontology as _module
from kgfoundry._namespace_proxy import (
    namespace_attach,
    namespace_dir,
    namespace_exports,
    namespace_getattr,
)

__all__ = namespace_exports(_module)
namespace_attach(_module, globals(), __all__)

__doc__ = _module.__doc__
if hasattr(_module, "__path__"):
    __path__ = list(_module.__path__)


def __getattr__(name: str) -> object:
    """Document   getattr  .

    &lt;!-- auto:docstring-builder v1 --&gt;

    Provide a fallback for unknown attribute lookups. This special method integrates the class with Python&#39;s data model so instances behave consistently with the language expectations.

    Parameters
    ----------
    name : str
        Configure the name.


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
