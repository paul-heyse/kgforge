"""Module for kgfoundry."""

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
    """Import the real module and register it under the kgfoundry namespace."""
    import importlib
    import sys

    module = importlib.import_module(_ALIASES[name])
    sys.modules[f"{__name__}.{name}"] = module
    return module


def __getattr__(name: str) -> object:
    """Load a lazily requested submodule from the compatibility namespace."""
    if name not in _ALIASES:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message) from None
    return _load(name)


def __dir__() -> list[str]:
    """Expose the known compatibility submodules for auto-complete."""
    return sorted(set(__all__))


def _ensure_namespace_alias(name: str) -> None:
    """Guarantee that ``kgfoundry.<name>`` resolves to the real package."""
    import sys

    if f"{__name__}.{name}" not in sys.modules:
        _load(name)


for _name in __all__:
    _ensure_namespace_alias(_name)
