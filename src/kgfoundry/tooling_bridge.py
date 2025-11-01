"""Public adapters for bridging tooling namespaces into the ``kgfoundry`` package.

These helpers delegate to the internal :mod:`kgfoundry._namespace_proxy` module while
providing typed, documented entry points for downstream packages. They are the
supported way to expose third-party modules (for example, ``search_client`` or
``vectorstore_faiss``) under the ``kgfoundry`` namespace.

Install the ``kgfoundry[tools]`` optional extra to ensure the tooling package is
available in the current environment before importing this module.
"""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from types import ModuleType

from kgfoundry._namespace_proxy import (
    namespace_attach as _namespace_attach,
)
from kgfoundry._namespace_proxy import (
    namespace_dir as _namespace_dir,
)
from kgfoundry._namespace_proxy import (
    namespace_exports as _namespace_exports,
)
from kgfoundry._namespace_proxy import (
    namespace_getattr as _namespace_getattr,
)

__all__ = [
    "namespace_attach",
    "namespace_dir",
    "namespace_exports",
    "namespace_getattr",
]


def namespace_attach(
    module: ModuleType,
    target: MutableMapping[str, object],
    names: Iterable[str],
) -> None:
    """Populate ``target`` with ``names`` sourced from ``module``.

    Parameters
    ----------
    module : ModuleType
        The module whose attributes are being proxied (for example, a
        third-party package).
    target : MutableMapping[str, object]
        The namespace to populate, typically ``globals()`` of the bridge module.
    names : Iterable[str]
        The attribute names that should be exposed publicly.
    """
    _namespace_attach(module, target, names)


def namespace_exports(module: ModuleType) -> list[str]:
    """Return the public export list for ``module``.

    The helper respects ``__all__`` when present and otherwise derives a sensible
    default by filtering out private attributes.
    """
    return _namespace_exports(module)


def namespace_dir(module: ModuleType, exports: Iterable[str]) -> list[str]:
    """Compose the ``dir()`` listing for a proxied module.

    Parameters
    ----------
    module : ModuleType
        The module whose attributes are being surfaced.
    exports : Iterable[str]
        The attribute names explicitly exposed by the bridge.

    Returns
    -------
    list[str]
        A sorted list of attribute names that should appear in ``dir()`` output.
    """
    return _namespace_dir(module, exports)


def namespace_getattr(module: ModuleType, name: str) -> object:
    """Resolve ``name`` from ``module`` while preserving the original attribute."""
    return _namespace_getattr(module, name)
