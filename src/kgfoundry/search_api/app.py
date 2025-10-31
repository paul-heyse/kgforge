"""Expose ``search_api.app`` inside the ``kgfoundry`` namespace."""

from __future__ import annotations

from typing import cast

import search_api.app as _module
from kgfoundry._namespace_proxy import (
    namespace_attach,
    namespace_dir,
    namespace_exports,
    namespace_getattr,
)

__all__ = namespace_exports(_module)
namespace_attach(_module, cast(dict[str, object], globals()), __all__)

__doc__ = _module.__doc__
if hasattr(_module, "__path__"):
    __path__ = list(_module.__path__)


def __getattr__(name: str) -> object:
    """Delegate dynamic attribute lookup to the implementation module."""
    return namespace_getattr(_module, name)


def __dir__() -> list[str]:
    """Return the combined attribute listing."""
    return namespace_dir(_module, __all__)
