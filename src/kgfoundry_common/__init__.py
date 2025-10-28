"""Kgfoundry Common utilities."""

from kgfoundry_common import (
    config,
    errors,
    exceptions,
    ids,
    logging,
    models,
    navmap_types,
    parquet_io,
)
from kgfoundry_common.navmap_types import NavMap

__all__ = [
    "config",
    "errors",
    "exceptions",
    "ids",
    "logging",
    "models",
    "navmap_types",
    "parquet_io",
]

__navmap__: NavMap = {
    "title": "kgfoundry_common",
    "synopsis": "Shared utilities and data structures used across kgfoundry",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@kgfoundry-common",
        "stability": "stable",
        "since": "0.1.0",
    },
    "symbols": {
        name: {
            "owner": "@kgfoundry-common",
            "stability": "stable",
            "since": "0.1.0",
        }
        for name in __all__
    },
}
