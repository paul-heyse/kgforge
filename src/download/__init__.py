"""Overview of download.

This module bundles download logic for the kgfoundry stack. It groups
related helpers so downstream packages can import a single cohesive
namespace. Refer to the functions and classes below for implementation
specifics.
"""

from download import cli, harvester

# [nav:anchor cli]
# [nav:anchor harvester]
from kgfoundry_common.navmap_types import NavMap

__all__ = ["cli", "harvester"]

__navmap__: NavMap = {
    "title": "download",
    "synopsis": "Public modules for the download and harvesting pipeline",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@download",
        "stability": "experimental",
        "since": "0.1.0",
    },
    "symbols": {
        "cli": {
            "stability": "beta",
            "owner": "@download",
            "since": "0.1.0",
        },
        "harvester": {
            "stability": "experimental",
            "owner": "@download",
            "since": "0.1.0",
        },
    },
}
