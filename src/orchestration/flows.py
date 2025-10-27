"""Module for orchestration.flows.

NavMap:
- NavMap: Structure describing a module navmap.
- t_echo: Return ``msg`` unmodified; handy for flow scaffolding and….
- e2e_flow: Run the high-level kgfoundry demo flow and return step names.
"""

from __future__ import annotations

from typing import Final

from prefect import flow, task

from kgfoundry_common.navmap_types import NavMap

__all__ = ["e2e_flow", "t_echo"]

__navmap__: Final[NavMap] = {
    "title": "orchestration.flows",
    "synopsis": "Prefect orchestration flows used in kgfoundry demos",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": ["t_echo", "e2e_flow"],
        },
    ],
}


# [nav:anchor t_echo]
@task
def t_echo(msg: str) -> str:
    """Return ``msg`` unmodified; handy for flow scaffolding and tests."""
    return msg


# [nav:anchor e2e_flow]
@flow(name="kgfoundry_e2e_skeleton")
def e2e_flow() -> list[str]:
    """Run the high-level kgfoundry demo flow and return step names."""
    return [
        t_echo.submit(x).result()
        for x in [
            "harvest",
            "doctags",
            "chunk",
            "embed_dense",
            "encode_splade",
            "bm25",
            "faiss",
            "ontology",
            "concept_embed",
            "linker",
            "kg",
        ]
    ]
