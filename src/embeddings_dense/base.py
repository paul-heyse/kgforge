"""Module for embeddings_dense.base.

NavMap:
- NavMap: Structure describing a module navmap.
- DenseEmbeddingModel: Protocol describing dense embedding encoders used in….
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Protocol

import numpy as np
from numpy.typing import NDArray

from kgfoundry_common.navmap_types import NavMap

__all__ = ["DenseEmbeddingModel"]

__navmap__: Final[NavMap] = {
    "title": "embeddings_dense.base",
    "synopsis": "Protocols describing dense embedding providers",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": ["DenseEmbeddingModel"],
        },
    ],
}


# [nav:anchor DenseEmbeddingModel]
class DenseEmbeddingModel(Protocol):
    """Protocol describing dense embedding encoders used in kgfoundry."""

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Encode ``texts`` into a dense floating-point representation."""
        ...
