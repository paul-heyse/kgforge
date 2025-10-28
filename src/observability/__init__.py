"""Observability utilities."""

# [nav:anchor metrics]
from kgfoundry_common.navmap_types import NavMap
from observability import metrics

__all__ = ["metrics"]

__navmap__: NavMap = {
    "title": "observability",
    "synopsis": "Observability metrics and instrumentation helpers",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@observability",
        "stability": "experimental",
        "since": "0.1.0",
    },
    "symbols": {
        "metrics": {
            "stability": "beta",
            "owner": "@observability",
            "since": "0.1.0",
        },
    },
}
