"""Expose ``embeddings_sparse.base`` inside the ``kgfoundry`` namespace."""

from __future__ import annotations

from embeddings_sparse.base import SparseEncoder, SparseIndex

__all__ = ["SparseEncoder", "SparseIndex"]
