"""Overview of orchestration.

This module bundles orchestration logic for the kgfoundry stack. It groups related helpers so
downstream packages can import a single cohesive namespace. Refer to the functions and classes below
for implementation specifics.
"""

from kgfoundry_common.navmap_types import NavMap
from orchestration import cli, fixture_flow, flows

__all__ = ["cli", "fixture_flow", "flows"]

__navmap__: NavMap = {
    "title": "orchestration",
    "synopsis": "Prefect flows and CLI entrypoints for kgfoundry orchestrations",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@orchestration",
        "stability": "experimental",
        "since": "0.1.0",
    },
    "symbols": {
        "cli": {
            "stability": "beta",
            "owner": "@orchestration",
            "since": "0.1.0",
        },
        "fixture_flow": {
            "stability": "experimental",
            "owner": "@orchestration",
            "since": "0.1.0",
        },
        "flows": {
            "stability": "experimental",
            "owner": "@orchestration",
            "since": "0.1.0",
        },
    },
}

# [nav:anchor cli]
# [nav:anchor fixture_flow]
# [nav:anchor flows]
