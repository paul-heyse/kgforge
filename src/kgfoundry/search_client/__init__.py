"""Namespace bridge that exposes the ``search_client`` package under ``kgfoundry``."""

from __future__ import annotations

from typing import Any

import search_client as _module
from search_client import KGFoundryClient as _KGFoundryClient

KGFoundryClient = _KGFoundryClient

__all__ = ["KGFoundryClient"]
__doc__ = _module.__doc__
__path__ = list(_module.__path__)


def __getattr__(name: str) -> Any:
    return getattr(_module, name)


def __dir__() -> list[str]:
    candidates = set(__all__)
    candidates.update(name for name in dir(_module) if not name.startswith("__"))
    return sorted(candidates)
