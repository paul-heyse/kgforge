"""Regression tests for FAISS module adaptation helpers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from search_api.types import FaissIndexProtocol, VectorArray, wrap_faiss_module


class _FakeIndex:
    def __init__(self, label: str) -> None:
        self.label = label


class _LegacyFaissModule:
    METRIC_INNER_PRODUCT = 0
    METRIC_L2 = 1

    def __init__(self) -> None:
        self.index_factory_calls: list[tuple[int, str, int]] = []
        self.write_calls: list[tuple[FaissIndexProtocol, str]] = []
        self.read_path: str | None = None
        self.normalize_vectors: VectorArray | None = None
        self.index_flat_ip_dimension: int | None = None
        self.index_id_map2_index: FaissIndexProtocol | None = None

    def index_factory(self, dimension: int, factory_string: str, metric: int) -> FaissIndexProtocol:
        self.index_factory_calls.append((dimension, factory_string, metric))
        return _FakeIndex("factory")

    def write_index(self, index: FaissIndexProtocol, path: str) -> None:
        self.write_calls.append((index, path))

    def read_index(self, path: str) -> FaissIndexProtocol:
        self.read_path = path
        return _FakeIndex("read")

    def IndexFlatIP(
        self, dimension: int
    ) -> FaissIndexProtocol:  # pragma: no cover - attribute accessed via adapter
        self.index_flat_ip_dimension = dimension
        return _FakeIndex("flat")

    def IndexIDMap2(
        self, index: FaissIndexProtocol
    ) -> FaissIndexProtocol:  # pragma: no cover - attribute accessed via adapter
        self.index_id_map2_index = index
        return _FakeIndex("idmap")

    def normalize_L2(
        self, vectors: VectorArray
    ) -> None:  # pragma: no cover - attribute accessed via adapter
        self.normalize_vectors = vectors


class _ModernFaissModule:
    METRIC_INNER_PRODUCT = 2
    METRIC_L2 = 3

    def __init__(self, builder: Callable[..., FaissIndexProtocol]) -> None:
        self._builder = builder

    def index_flat_ip(self, dimension: int) -> FaissIndexProtocol:
        return self._builder("flat", dimension)

    def index_factory(self, dimension: int, factory_string: str, metric: int) -> FaissIndexProtocol:
        return self._builder("factory", dimension)

    def index_id_map2(self, index: FaissIndexProtocol) -> FaissIndexProtocol:
        return self._builder("idmap", index)

    def write_index(self, index: FaissIndexProtocol, path: str) -> None:
        self._builder("write", path)

    def read_index(self, path: str) -> FaissIndexProtocol:
        return self._builder("read", path)

    def normalize_l2(self, vectors: VectorArray) -> None:
        self._builder("normalize", vectors)


@pytest.mark.parametrize("dimension", [5, 128])
def test_wrap_faiss_module_adapts_legacy_surface(dimension: int) -> None:
    legacy = _LegacyFaissModule()
    adapted = wrap_faiss_module(legacy)

    # Adapter exposes the modern PEP-8 surface while dispatching to legacy functions.
    index = adapted.index_flat_ip(dimension)
    assert isinstance(index, _FakeIndex)
    assert legacy.index_flat_ip_dimension == dimension

    mapped_index = adapted.index_id_map2(index)
    assert isinstance(mapped_index, _FakeIndex)
    assert legacy.index_id_map2_index is index

    vectors = np.ones((2, dimension), dtype=np.float32)
    adapted.normalize_l2(vectors)
    assert legacy.normalize_vectors is not None
    assert legacy.normalize_vectors.shape == vectors.shape

    factory_index = adapted.index_factory(dimension, "Flat", legacy.metric_l2)
    assert isinstance(factory_index, _FakeIndex)
    assert legacy.index_factory_calls[-1] == (dimension, "Flat", legacy.metric_l2)

    adapted.write_index(factory_index, "/tmp/index.faiss")
    assert legacy.write_calls[-1] == (factory_index, "/tmp/index.faiss")

    read_index = adapted.read_index("/tmp/index.faiss")
    assert isinstance(read_index, _FakeIndex)
    assert legacy.read_path == "/tmp/index.faiss"


def test_wrap_faiss_module_returns_pep8_module_directly() -> None:
    records: list[tuple[str, object]] = []

    def builder(label: str, value: object) -> FaissIndexProtocol:
        records.append((label, value))
        return _FakeIndex(label)

    modern = _ModernFaissModule(builder)
    wrapped = wrap_faiss_module(modern)

    # The helper should return the module unchanged when it already satisfies the protocol.
    assert wrapped is modern

    result = wrapped.index_flat_ip(10)
    assert isinstance(result, _FakeIndex)
    assert records[0] == ("flat", 10)
