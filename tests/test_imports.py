"""Validate that namespaced kgfoundry imports resolve correctly."""

from __future__ import annotations

import importlib


def test_imports() -> None:
    module = importlib.import_module("kgfoundry.kgfoundry_common.models")
    assert hasattr(module, "Doc")


def test_namespace_bridges() -> None:
    cases = {
        "kgfoundry.embeddings_sparse.bm25": "get_bm25",
        "kgfoundry.embeddings_sparse.splade": "get_splade",
        "kgfoundry.kg_builder.mock_kg": "MockKG",
        "kgfoundry.registry.helper": "DuckDBRegistryHelper",
        "kgfoundry.registry.migrate": "apply",
        "kgfoundry.search_api.app": "search",
        "kgfoundry.search_client": "KGFoundryClient",
    }

    for path, attribute in cases.items():
        module = importlib.import_module(path)
        assert hasattr(module, attribute), f"{path} missing {attribute}"
