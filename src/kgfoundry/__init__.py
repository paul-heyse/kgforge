"""Overview of kgfoundry.

This module bundles kgfoundry logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
"""

from __future__ import annotations

import sys
from importlib import import_module
from typing import TYPE_CHECKING

_ALIASES: dict[str, str] = {
    "docling": "kgfoundry.docling",
    "download": "kgfoundry.download",
    "embeddings_dense": "kgfoundry.embeddings_dense",
    "embeddings_sparse": "kgfoundry.embeddings_sparse",
    "kg_builder": "kgfoundry.kg_builder",
    "kgfoundry_common": "kgfoundry_common",
    "linking": "kgfoundry.linking",
    "observability": "kgfoundry.observability",
    "ontology": "kgfoundry.ontology",
    "orchestration": "kgfoundry.orchestration",
    "registry": "kgfoundry.registry",
    "search_api": "kgfoundry.search_api",
    "search_client": "kgfoundry.search_client",
    "vectorstore_faiss": "kgfoundry.vectorstore_faiss",
}

__all__: list[str] = [
    "docling",
    "download",
    "embeddings_dense",
    "embeddings_sparse",
    "kg_builder",
    "kgfoundry_common",
    "linking",
    "observability",
    "ontology",
    "orchestration",
    "registry",
    "search_api",
    "search_client",
    "vectorstore_faiss",
]

if __debug__:
    expected: set[str] = set(_ALIASES)
    configured: set[str] = set(__all__)
    if expected != configured:
        missing = sorted(expected.difference(configured))
        extra = sorted(configured.difference(expected))
        message = f"kgfoundry exports mismatch. missing={missing!r} extra={extra!r}"
        raise RuntimeError(message)

if TYPE_CHECKING:
    from types import ModuleType

    kgfoundry_common: ModuleType = import_module("kgfoundry_common")
    docling: ModuleType = import_module("kgfoundry.docling")
    download: ModuleType = import_module("kgfoundry.download")
    embeddings_dense: ModuleType = import_module("kgfoundry.embeddings_dense")
    embeddings_sparse: ModuleType = import_module("kgfoundry.embeddings_sparse")
    kg_builder: ModuleType = import_module("kgfoundry.kg_builder")
    linking: ModuleType = import_module("kgfoundry.linking")
    observability: ModuleType = import_module("kgfoundry.observability")
    ontology: ModuleType = import_module("kgfoundry.ontology")
    orchestration: ModuleType = import_module("kgfoundry.orchestration")
    registry: ModuleType = import_module("kgfoundry.registry")
    search_api: ModuleType = import_module("kgfoundry.search_api")
    search_client: ModuleType = import_module("kgfoundry.search_client")
    vectorstore_faiss: ModuleType = import_module("kgfoundry.vectorstore_faiss")


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
