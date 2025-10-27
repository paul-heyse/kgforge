"""Module for embeddings_dense.base."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class DenseEmbedder(Protocol):
    """Protocol describing a dense embedding provider."""

    name: str
    dim: int

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Return embeddings for the supplied texts."""

        ...
