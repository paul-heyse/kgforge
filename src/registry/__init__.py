"""Registry utilities."""

import registry.api as api
import registry.duckdb_registry as duckdb_registry
import registry.helper as helper
import registry.migrate as migrate

from kgfoundry_common.navmap_types import NavMap

__all__ = ["api", "duckdb_registry", "helper", "migrate"]

__navmap__: NavMap = {
    "title": "registry",
    "synopsis": "Registry abstractions and DuckDB-backed implementations",
    "exports": __all__,
    "owner": "@registry",
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
        "api": {},
        "duckdb_registry": {},
        "helper": {},
        "migrate": {},
    },
}
