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
    "owner": "@kgfoundry-common",
    "stability": "experimental",
    "since": "0.1.0",
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "symbols": {name: {} for name in __all__},
}
