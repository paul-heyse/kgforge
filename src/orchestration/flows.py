"""Flows utilities."""

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
    """Return t echo.

    Parameters
    ----------
    msg : str
        Description for ``msg``.

    Returns
    -------
    str
        Description of return value.
    """
    return msg


# [nav:anchor e2e_flow]
@flow(name="kgfoundry_e2e_skeleton")
def e2e_flow() -> list[str]:
    """Return e2e flow.

    Returns
    -------
    List[str]
        Description of return value.
    """
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
