"""Module for embeddings_dense.base.

NavMap:
- DenseEmbedder: Protocol describing a dense embedding provider.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class DenseEmbedder(Protocol):
    """Protocol describing a dense embedding provider."""

    name: str
    dim: int

    def embed_texts(self, texts: list[str]) -> NDArray[np.float32]:
        """Return embeddings for the supplied texts."""
        ...
