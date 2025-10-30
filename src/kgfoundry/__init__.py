"""Overview of kgfoundry.

This module bundles kgfoundry logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
"""

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

__all__ = sorted(_ALIASES)


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
    import importlib
    import sys

    module = importlib.import_module(_ALIASES[name])
    sys.modules[f"{__name__}.{name}"] = module
    return module


def __getattr__(name: str) -> object:
    """Document   getattr  .

    <!-- auto:docstring-builder v1 -->

    Provide a fallback for unknown attribute lookups. This special method integrates the class with Python's data model so instances behave consistently with the language expectations.

    Parameters
    ----------
    name : str
        Describe `name`.
        
        

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
    import sys

    if f"{__name__}.{name}" not in sys.modules:
        _load(name)


for _name in __all__:
    _ensure_namespace_alias(_name)
