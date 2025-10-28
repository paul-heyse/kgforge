"""Provide utilities for module.

Notes
-----
This module exposes the primary interfaces for the package.

See Also
--------
search_client
"""


from kgfoundry_common.navmap_types import NavMap
from search_client.client import KGFoundryClient as _KGFoundryClient

__all__ = ["KGFoundryClient"]

__navmap__: NavMap = {
    "title": "search_client.__init__",
    "synopsis": "Module for search_client",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": ["KGFoundryClient"],
        },
    ],
}


# [nav:anchor KGFoundryClient]
KGFoundryClient = _KGFoundryClient
