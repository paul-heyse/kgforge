"""Overview of kgfoundry common.

This module bundles kgfoundry common logic for the kgfoundry stack. It groups related helpers so
downstream packages can import a single cohesive namespace. Refer to the functions and classes below
for implementation specifics.
"""

# [nav:anchor config]
# [nav:anchor errors]
# [nav:anchor exceptions]
# [nav:anchor fs]
# [nav:anchor ids]
# [nav:anchor logging]
# [nav:anchor models]
# [nav:anchor navmap_types]
# [nav:anchor parquet_io]
# [nav:anchor safe_pickle_v2]
# [nav:anchor subprocess_utils]
from kgfoundry_common import (
    config,
    errors,
    exceptions,
    fs,
    ids,
    logging,
    models,
    navmap_types,
    observability,
    parquet_io,
    safe_pickle_v2,
    schema_helpers,
    sequence_guards,
    serialization,
    settings,
    subprocess_utils,
    vector_types,
)
from kgfoundry_common.navmap_types import NavMap

__all__ = [
    "config",
    "errors",
    "exceptions",
    "fs",
    "ids",
    "logging",
    "models",
    "navmap_types",
    "observability",
    "parquet_io",
    "safe_pickle_v2",
    "schema_helpers",
    "sequence_guards",
    "serialization",
    "settings",
    "subprocess_utils",
    "vector_types",
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
