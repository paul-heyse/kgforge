"""Kgfoundry utilities."""

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
    """Compute load.

    Carry out the load operation.

    Parameters
    ----------
    name : str
        Description for ``name``.

    Returns
    -------
    object
        Description of return value.
    """
    import importlib
    import sys

    module = importlib.import_module(_ALIASES[name])
    sys.modules[f"{__name__}.{name}"] = module
    return module


def __getattr__(name: str) -> object:
    """Compute getattr.

    Carry out the getattr operation.

    Parameters
    ----------
    name : str
        Description for ``name``.

    Returns
    -------
    object
        Description of return value.

    Raises
    ------
    AttributeError
        Raised when validation fails.
    """
    if name not in _ALIASES:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message) from None
    return _load(name)


def __dir__() -> list[str]:
    """Compute dir.

    Carry out the dir operation.

    Returns
    -------
    List[str]
        Description of return value.
    """
    return sorted(set(__all__))


def _ensure_namespace_alias(name: str) -> None:
    """Compute ensure namespace alias.

    Carry out the ensure namespace alias operation.

    Parameters
    ----------
    name : str
        Description for ``name``.
    """
    import sys

    if f"{__name__}.{name}" not in sys.modules:
        _load(name)


for _name in __all__:
    _ensure_namespace_alias(_name)
