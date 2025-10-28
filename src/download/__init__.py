"""Download utilities."""

import download.cli as cli
import download.harvester as harvester

from kgfoundry_common.navmap_types import NavMap

__all__ = ["cli", "harvester"]

__navmap__: NavMap = {
    "title": "download",
    "synopsis": "Public modules for the download and harvesting pipeline",
    "exports": __all__,
    "owner": "@download",
    "stability": "experimental",
    "since": "0.1.0",
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "symbols": {
        "cli": {},
        "harvester": {},
    },
}
