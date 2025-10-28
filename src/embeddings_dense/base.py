"""Base utilities."""

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
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@embeddings",
        "stability": "beta",
        "since": "0.1.0",
    },
    "symbols": {
        "DenseEmbeddingModel": {
            "owner": "@embeddings",
            "stability": "beta",
            "since": "0.1.0",
        },
    },
}


# [nav:anchor DenseEmbeddingModel]
class DenseEmbeddingModel(Protocol):
    """Describe DenseEmbeddingModel."""

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Compute encode.

        Carry out the encode operation.

        Parameters
        ----------
        texts : collections.abc.Sequence
            Description for ``texts``.

        Returns
        -------
        numpy.typing.NDArray
            Description of return value.

        Examples
        --------
        >>> from embeddings_dense.base import encode
        >>> result = encode(...)
        >>> result  # doctest: +ELLIPSIS
        ...
        """
        ...
