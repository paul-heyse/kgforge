"""Public helpers for kgfoundry namespace bridge packages."""

from __future__ import annotations

from kgfoundry.tooling_bridge import (
    namespace_attach,
    namespace_dir,
    namespace_exports,
    namespace_getattr,
)

__all__ = [
    "namespace_attach",
    "namespace_dir",
    "namespace_exports",
    "namespace_getattr",
]
