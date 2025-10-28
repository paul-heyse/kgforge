"""Overview of embeddings dense.

This module bundles embeddings dense logic for the kgfoundry stack. It groups related helpers so
downstream packages can import a single cohesive namespace. Refer to the functions and classes below
for implementation specifics.
"""


from embeddings_dense import base, qwen3
from kgfoundry_common.navmap_types import NavMap

__all__ = ["base", "qwen3"]

__navmap__: NavMap = {
    "title": "embeddings_dense",
    "synopsis": "Dense embedding adapters and protocols",
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
        "stability": "experimental",
        "since": "0.1.0",
    },
    "symbols": {
        "base": {
            "stability": "beta",
            "owner": "@embeddings",
            "since": "0.1.0",
        },
        "qwen3": {
            "stability": "experimental",
            "owner": "@embeddings",
            "since": "0.2.0",
        },
    },
}

# [nav:anchor base]
# [nav:anchor qwen3]
