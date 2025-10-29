"""Overview of search client.

This module bundles search client logic for the kgfoundry stack. It
groups related helpers so downstream packages can import a single
cohesive namespace. Refer to the functions and classes below for
implementation specifics.
"""

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
        "KGFoundryClient": {
            "stability": "experimental",
            "owner": "@search-api",
            "since": "0.2.0",
        },
    },
}


# [nav:anchor KGFoundryClient]
KGFoundryClient = _KGFoundryClient
