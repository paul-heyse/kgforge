"""Expose ``embeddings_sparse.splade`` inside the ``kgfoundry`` namespace."""

from __future__ import annotations

from embeddings_sparse.splade import (
    LuceneImpactIndex,
    PureImpactIndex,
    SPLADEv3Encoder,
    get_splade,
)

__all__ = [
    "LuceneImpactIndex",
    "PureImpactIndex",
    "SPLADEv3Encoder",
    "get_splade",
]
