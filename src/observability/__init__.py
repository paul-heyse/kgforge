"""Observability utilities."""

from typing import Final

import observability.metrics as metrics

from kgfoundry_common.navmap_types import NavMap

__all__ = ["metrics"]

__navmap__: Final[NavMap] = {
    "title": "observability",
    "synopsis": "Metrics and tracing utilities for kgfoundry services",
    "exports": __all__,
    "owner": "@observability",
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
        "metrics": {},
    },
}
