"""Linking utilities."""

from kgfoundry_common.navmap_types import NavMap
from linking import calibration, linker

__all__ = ["calibration", "linker"]

__navmap__: NavMap = {
    "title": "linking",
    "synopsis": "Entity linking calibration and production pipelines",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@linking",
        "stability": "experimental",
        "since": "0.1.0",
    },
    "symbols": {
        "calibration": {
            "stability": "beta",
            "owner": "@linking",
            "since": "0.1.0",
        },
        "linker": {
            "stability": "beta",
            "owner": "@linking",
            "since": "0.1.0",
        },
    },
}

# [nav:anchor calibration]
# [nav:anchor linker]
