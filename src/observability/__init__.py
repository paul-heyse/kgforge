"""Observability utilities."""

# [nav:anchor metrics]
from kgfoundry_common.navmap_types import NavMap
from observability import metrics

__all__ = ["metrics"]

__navmap__: NavMap = {
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
        "metrics": {
            "stability": "beta",
            "owner": "@observability",
            "since": "0.1.0",
        },
    },
}
