"""Overview of kgfoundry.

This module bundles kgfoundry logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
"""

from __future__ import annotations

import sys
from importlib import import_module
from typing import TYPE_CHECKING, Dict, List

_ALIASES: dict[str, str] = {
    "docling": "docling",
    "download": "download",
    "embeddings_dense": "embeddings_dense",
    "embeddings_sparse": "embeddings_sparse",
    "kg_builder": "kg_builder",
    "kgfoundry_common": "kgfoundry_common",
    "linking": "linking",
    "observability": "observability",
    "ontology": "ontology",
    "orchestration": "orchestration",
    "registry": "registry",
    "search_api": "search_api",
    "search_client": "search_client",
    "vectorstore_faiss": "vectorstore_faiss",
}

if TYPE_CHECKING:
    from . import docling as docling
    from . import download as download
    from . import embeddings_dense as embeddings_dense
    from . import embeddings_sparse as embeddings_sparse
    from . import kg_builder as kg_builder
    from . import kgfoundry_common as kgfoundry_common
    from . import linking as linking
    from . import observability as observability
    from . import ontology as ontology
    from . import orchestration as orchestration
    from . import registry as registry
    from . import search_api as search_api
    from . import search_client as search_client
    from . import vectorstore_faiss as vectorstore_faiss

__all__: list[str] = sorted(_ALIASES)


def _load(name: str) -> object:
    """Document  load.

    <!-- auto:docstring-builder v1 -->

    Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

    Parameters
    ----------
    name : str
        Describe ``name``.

    Returns
    -------
    object
        Describe return value.
    """
    module = import_module(_ALIASES[name])
    sys.modules[f"{__name__}.{name}"] = module
    return module


def __getattr__(name: str) -> object:
    """Document   getattr  .

    <!-- auto:docstring-builder v1 -->

    &lt;!-- auto:docstring-builder v1 --&gt;

    Provide a fallback for unknown attribute lookups. This special method integrates the class with Python&#39;s data model so instances behave consistently with the language expectations.

    Parameters
    ----------
    name : str
        Configure the name.

    Returns
    -------
    object
        Describe return value.
    """
    if name not in _ALIASES:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message) from None
    return _load(name)


def __dir__() -> list[str]:
    """Return the combined attribute listing.

    <!-- auto:docstring-builder v1 -->

    Returns
    -------
    list[str]
        Sorted union of exports and implementation attributes.
    """
    return sorted(set(__all__))


def _ensure_namespace_alias(name: str) -> None:
    """Document  ensure namespace alias.

    <!-- auto:docstring-builder v1 -->

    Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

    Parameters
    ----------
    name : str
        Describe ``name``.
    """
    if f"{__name__}.{name}" not in sys.modules:
        _load(name)
