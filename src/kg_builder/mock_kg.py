"""Overview of mock kg.

This module bundles mock kg logic for the kgfoundry stack. It groups
related helpers so downstream packages can import a single cohesive
namespace. Refer to the functions and classes below for implementation
specifics.
"""

from __future__ import annotations

from typing import Final

from kgfoundry_common.navmap_types import NavMap

__all__ = ["MockKG"]

__navmap__: Final[NavMap] = {
    "title": "kg_builder.mock_kg",
    "synopsis": "Helpers for the MockKG in-memory knowledge graph",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@kg-builder",
        "stability": "experimental",
        "since": "0.1.0",
    },
    "symbols": {
        "MockKG": {
            "owner": "@kg-builder",
            "stability": "experimental",
            "since": "0.1.0",
        },
    },
}


# [nav:anchor MockKG]
class MockKG:
    """Describe MockKG.

<!-- auto:docstring-builder v1 -->

how instances collaborate with the surrounding package. Highlight
how the class supports nearby modules to guide readers through the
codebase.
"""

    def __init__(self) -> None:
        """Describe   init  .

<!-- auto:docstring-builder v1 -->

Python's object protocol for this class. Use it to integrate
with built-in operators, protocols, or runtime behaviours that
expect instances to participate in the language's data model.
"""
        self.chunk2concepts: dict[str, set[str]] = {}
        self.neighbors: dict[str, set[str]] = {}

    def add_mention(self, chunk_id: str, concept_id: str) -> None:
        """Describe add mention.

<!-- auto:docstring-builder v1 -->

Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

Parameters
----------
chunk_id : str
    Describe ``chunk_id``.
concept_id : str
    Describe ``concept_id``.
"""
        self.chunk2concepts.setdefault(chunk_id, set()).add(concept_id)

    def add_edge(self, a: str, b: str) -> None:
        """Describe add edge.

<!-- auto:docstring-builder v1 -->

Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

Parameters
----------
a : str
    Describe ``a``.
b : str
    Describe ``b``.
"""
        self.neighbors.setdefault(a, set()).add(b)
        self.neighbors.setdefault(b, set()).add(a)

    def linked_concepts(self, chunk_id: str) -> list[str]:
        """Describe linked concepts.

<!-- auto:docstring-builder v1 -->

Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

Parameters
----------
chunk_id : str
    Describe ``chunk_id``.

Returns
-------
list[str]
    Describe return value.
"""
        return sorted(self.chunk2concepts.get(chunk_id, set()))

    def one_hop(self, concept_id: str) -> list[str]:
        """Describe one hop.

<!-- auto:docstring-builder v1 -->

Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

Parameters
----------
concept_id : str
    Describe ``concept_id``.

Returns
-------
list[str]
    Describe return value.
"""
        return sorted(self.neighbors.get(concept_id, set()))
