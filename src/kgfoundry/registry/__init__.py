"""Namespace bridge that exposes the registry package under kgfoundry."""

from __future__ import annotations

from typing import Any

import registry as _module

__all__ = list(getattr(_module, "__all__", []))
if not __all__:
    __all__ = [name for name in dir(_module) if not name.startswith("_")]
for _name in __all__:
    globals()[_name] = getattr(_module, _name)

__doc__ = _module.__doc__
if hasattr(_module, "__path__"):
    __path__ = list(_module.__path__)


def __getattr__(name: str) -> Any:
    return getattr(_module, name)


def __dir__() -> list[str]:
    candidates = set(__all__)
    candidates.update(name for name in dir(_module) if not name.startswith("__"))
    return sorted(candidates)
