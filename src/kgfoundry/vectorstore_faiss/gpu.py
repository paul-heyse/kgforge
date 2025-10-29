"""Expose ``vectorstore_faiss.gpu`` inside the ``kgfoundry`` namespace."""

from __future__ import annotations

from vectorstore_faiss.gpu import FaissGpuIndex, FloatArray, IntArray, StrArray

__all__ = ["FaissGpuIndex", "FloatArray", "IntArray", "StrArray"]
