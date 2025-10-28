"""Linking utilities."""

from kgfoundry_common.navmap_types import NavMap
from linking import calibration, linker

__all__ = ["calibration", "linker"]

__navmap__: NavMap = {
    "title": "linking",
    "synopsis": "Knowledge graph linking orchestration components",
    "exports": __all__,
    "owner": "@linking",
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
        "calibration": {
            "stability": "experimental",
            "owner": "@linking",
            "since": "0.1.0",
        },
        "linker": {
            "stability": "experimental",
            "owner": "@linking",
            "since": "0.1.0",
        },
    },
}

# [nav:anchor calibration]
# [nav:anchor linker]
