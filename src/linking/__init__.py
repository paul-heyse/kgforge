"""Linking utilities."""

from typing import Final

import linking.calibration as calibration
import linking.linker as linker

from kgfoundry_common.navmap_types import NavMap

__all__ = ["calibration", "linker"]

__navmap__: Final[NavMap] = {
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
        "calibration": {},
        "linker": {},
    },
}
