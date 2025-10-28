"""Registry utilities."""

from kgfoundry_common.navmap_types import NavMap
from registry import api, duckdb_registry, helper, migrate

__all__ = ["api", "duckdb_registry", "helper", "migrate"]

__navmap__: NavMap = {
    "title": "registry",
    "synopsis": "DuckDB-backed registry APIs and helpers",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@registry",
        "stability": "experimental",
        "since": "0.1.0",
    },
    "symbols": {
        "api": {
            "stability": "beta",
            "owner": "@registry",
            "since": "0.1.0",
        },
        "duckdb_registry": {
            "stability": "beta",
            "owner": "@registry",
            "since": "0.1.0",
        },
        "helper": {
            "stability": "beta",
            "owner": "@registry",
            "since": "0.1.0",
        },
        "migrate": {
            "stability": "experimental",
            "owner": "@registry",
            "since": "0.1.0",
        },
    },
}

# [nav:anchor api]
# [nav:anchor duckdb_registry]
# [nav:anchor helper]
# [nav:anchor migrate]
