from __future__ import annotations
from typing import Protocol, List
import numpy as np

class DenseEmbedder(Protocol):
    name: str
    dim: int
    def embed_texts(self, texts: List[str]) -> np.ndarray: ...
