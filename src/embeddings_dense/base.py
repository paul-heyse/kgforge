"""Overview of base.

This module bundles base logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
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
    """Model the DenseEmbeddingModel.
<!-- auto:docstring-builder v1 -->

Represent the denseembeddingmodel data structure used throughout the project. The class
encapsulates behaviour behind a well-defined interface for collaborating components. Instances
are typically created by factories or runtime orchestrators documented nearby.

Parameters
----------
*args : Any
    Describe ``args``.
**kwargs : Any
    Describe ``kwargs``.
"""

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Compute encode.
<!-- auto:docstring-builder v1 -->

Carry out the encode operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

Parameters
----------
texts : Sequence[str]
    Description for ``texts``.
    
    
    

Returns
-------
NDArray[np.float32]
    Description of return value.
    
    
    

Examples
--------
>>> from embeddings_dense.base import encode
>>> result = encode(...)
>>> result  # doctest: +ELLIPSIS
"""
        ...
