"""Registry utilities."""

from kgfoundry_common.navmap_types import NavMap
from registry import api, duckdb_registry, helper, migrate

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
        "api": {
            "stability": "beta",
            "owner": "@registry",
            "since": "0.1.0",
        },
        "duckdb_registry": {
            "stability": "experimental",
            "owner": "@registry",
            "since": "0.2.0",
        },
        "helper": {
            "stability": "experimental",
            "owner": "@registry",
            "since": "0.1.0",
        },
        "migrate": {
            "stability": "experimental",
            "owner": "@registry",
            "since": "0.2.0",
        },
    },
}

# [nav:anchor api]
# [nav:anchor duckdb_registry]
# [nav:anchor helper]
# [nav:anchor migrate]
