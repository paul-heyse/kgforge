"""Module for orchestration.flows.

NavMap:
- t_echo: T echo.
- e2e_flow: E2e flow.
"""

from __future__ import annotations

from prefect import flow, task


@task
def t_echo(msg: str) -> str:
    """T echo.

    Parameters
    ----------
    msg : str
        TODO.

    Returns
    -------
    str
        TODO.
    """
    return msg


@flow(name="kgfoundry_e2e_skeleton")
def e2e_flow() -> list[str]:
    """E2e flow."""
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
