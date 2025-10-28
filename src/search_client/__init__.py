"""Search Client utilities."""

from kgfoundry_common.navmap_types import NavMap
from search_client.client import KGFoundryClient as _KGFoundryClient

__all__ = ["KGFoundryClient"]

__navmap__: NavMap = {
    "title": "search_client",
    "synopsis": "Client abstractions for calling the kgfoundry Search API",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@search-api",
        "stability": "experimental",
        "since": "0.2.0",
    },
    "symbols": {
        "client": {
            "stability": "experimental",
            "owner": "@search-api",
            "since": "0.2.0",
        },
    },
}


# [nav:anchor KGFoundryClient]
KGFoundryClient = _KGFoundryClient
