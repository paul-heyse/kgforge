"""Orchestration utilities."""

from kgfoundry_common.navmap_types import NavMap
from orchestration import cli, fixture_flow, flows

__all__ = ["cli", "fixture_flow", "flows"]

__navmap__: NavMap = {
    "title": "orchestration",
    "synopsis": "Prefect flows and CLI entrypoints for kgfoundry orchestrations",
    "exports": __all__,
    "owner": "@orchestration",
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
