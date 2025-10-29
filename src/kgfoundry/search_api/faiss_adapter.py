"""Expose ``search_api.faiss_adapter`` inside the ``kgfoundry`` namespace."""

from __future__ import annotations

from typing import Any

import search_api.faiss_adapter as _module

from search_api.faiss_adapter import DenseVecs, FaissAdapter, VecArray, HAVE_FAISS

__all__ = ["DenseVecs", "FaissAdapter", "VecArray", "HAVE_FAISS"]
__doc__ = _module.__doc__
__path__ = list(getattr(_module, "__path__", []))


def __getattr__(name: str) -> Any:
    return getattr(_module, name)


def __dir__() -> list[str]:
    candidates = set(__all__)
    candidates.update(name for name in dir(_module) if not name.startswith("__"))
    return sorted(candidates)
