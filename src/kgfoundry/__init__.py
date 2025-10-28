"""Provide utilities for module.

Notes
-----
This module exposes the primary interfaces for the package.

See Also
--------
kgfoundry
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
    """Return load.

    Parameters
    ----------
    name : str
        Description for ``name``.
    
    Returns
    -------
    object
        Description of return value.
    
    Examples
    --------
    >>> from kgfoundry import _load
    >>> result = _load(...)
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    kgfoundry
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
    """
    
    import importlib
    import sys

    module = importlib.import_module(_ALIASES[name])
    sys.modules[f"{__name__}.{name}"] = module
    return module


def __getattr__(name: str) -> object:
    """Return getattr.

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
    
    Examples
    --------
    >>> from kgfoundry import __getattr__
    >>> result = __getattr__(...)
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    kgfoundry
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
    """
    
    if name not in _ALIASES:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message) from None
    return _load(name)


def __dir__() -> list[str]:
    """Return dir.

    Returns
    -------
    List[str]
        Description of return value.
    
    Examples
    --------
    >>> from kgfoundry import __dir__
    >>> result = __dir__()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    kgfoundry
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
    """
    
    
    return sorted(set(__all__))


def _ensure_namespace_alias(name: str) -> None:
    """Return ensure namespace alias.

    Parameters
    ----------
    name : str
        Description for ``name``.
    
    Examples
    --------
    >>> from kgfoundry import _ensure_namespace_alias
    >>> _ensure_namespace_alias(...)  # doctest: +ELLIPSIS
    
    See Also
    --------
    kgfoundry
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
    """
    
    import sys

    if f"{__name__}.{name}" not in sys.modules:
        _load(name)


for _name in __all__:
    _ensure_namespace_alias(_name)
