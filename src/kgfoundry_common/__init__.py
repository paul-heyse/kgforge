"""Kgfoundry Common utilities."""

import kgfoundry_common.config as config
import kgfoundry_common.errors as errors
import kgfoundry_common.exceptions as exceptions
import kgfoundry_common.ids as ids
import kgfoundry_common.logging as logging
import kgfoundry_common.models as models
import kgfoundry_common.navmap_types as navmap_types
import kgfoundry_common.parquet_io as parquet_io

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
