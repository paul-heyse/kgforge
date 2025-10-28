"""Orchestration utilities."""

from typing import Final

import orchestration.cli as cli
import orchestration.fixture_flow as fixture_flow
import orchestration.flows as flows

from kgfoundry_common.navmap_types import NavMap

__all__ = ["cli", "fixture_flow", "flows"]

__navmap__: Final[NavMap] = {
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
        "cli": {},
        "fixture_flow": {},
        "flows": {},
    },
}
