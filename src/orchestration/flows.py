"""Overview of flows.

This module bundles flows logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
"""

from __future__ import annotations

from typing import Final

from prefect import flow, task

from kgfoundry_common.navmap_types import NavMap

__all__ = ["e2e_flow", "t_echo"]

__navmap__: Final[NavMap] = {
    "title": "orchestration.flows",
    "synopsis": "Prefect flow definitions for kgfoundry pipelines",
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
        name: {
            "owner": "@orchestration",
            "stability": "experimental",
            "since": "0.1.0",
        }
        for name in __all__
    },
}


def _t_echo_impl(msg: str) -> str:
    """Compute t echo.

    Carry out the t echo operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    msg : str
        Description for ``msg``.

    Returns
    -------
    str
        Description of return value.

    Examples
    --------
    >>> from orchestration.flows import t_echo
    >>> result = t_echo(...)
    >>> result  # doctest: +ELLIPSIS
    """
    return msg


# [nav:anchor t_echo]
t_echo = task(_t_echo_impl)


def _e2e_flow_impl() -> list[str]:
    """Compute e2e flow.

    Carry out the e2e flow operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Returns
    -------
    List[str]
        Description of return value.

    Examples
    --------
    >>> from orchestration.flows import e2e_flow
    >>> result = e2e_flow()
    >>> result  # doctest: +ELLIPSIS
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


# [nav:anchor e2e_flow]
e2e_flow = flow(name="kgfoundry_e2e_skeleton")(_e2e_flow_impl)
