"""Namespace bridge that exposes the docling_kg package under kgfoundry."""
# [nav:section public-api]

from __future__ import annotations

from typing import cast

import docling_kg as _module
from kgfoundry.namespace_bridge import (
    namespace_attach,
    namespace_dir,
    namespace_exports,
    namespace_getattr,
)

_EXPORTS = tuple(namespace_exports(_module))
_namespace = cast("dict[str, object]", globals())
namespace_attach(_module, _namespace, _EXPORTS)

__doc__ = _module.__doc__
if hasattr(_module, "__path__"):
    __path__ = list(_module.__path__)


def __getattr__(name: str) -> object:
    """Forward attribute lookups to the underlying module."""
    return namespace_getattr(_module, name)


def __dir__() -> list[str]:
    """Return the combined attribute listing."""
    return namespace_dir(_module, _EXPORTS)
