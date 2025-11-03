"""Regression tests for FAISS module adaptation helpers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from search_api.types import FaissIndexProtocol, IndexArray, VectorArray, wrap_faiss_module


class _FakeIndex:
    def __init__(self, label: str) -> None:
        self.label = label
        self.last_added: VectorArray | None = None
        self.last_added_with_ids: tuple[VectorArray, IndexArray] | None = None
        self.last_trained: VectorArray | None = None

    def add(self, vectors: VectorArray) -> None:
        self.last_added = vectors

    def search(
        self, vectors: VectorArray, k: int
    ) -> tuple[
        npt.NDArray[np.float32], npt.NDArray[np.int64]
    ]:  # pragma: no cover - minimal protocol stub
        distances = np.zeros((vectors.shape[0], k), dtype=np.float32)
        indices = np.zeros((vectors.shape[0], k), dtype=np.int64)
        return distances, indices

    def train(self, vectors: VectorArray) -> None:  # pragma: no cover - optional protocol method
        self.last_trained = vectors

    def add_with_ids(
        self, vectors: VectorArray, ids: IndexArray
    ) -> None:  # pragma: no cover - optional protocol method
        self.last_added_with_ids = (vectors, ids)


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

        def index_flat_ip_impl(dimension: int) -> FaissIndexProtocol:
            self.index_flat_ip_dimension = dimension
            return _FakeIndex("flat")

        def index_id_map2_impl(index: FaissIndexProtocol) -> FaissIndexProtocol:
            self.index_id_map2_index = index
            return _FakeIndex("idmap")

        def normalize_l2_impl(vectors: VectorArray) -> None:
            self.normalize_vectors = vectors

        self.IndexFlatIP: Callable[[int], FaissIndexProtocol] = index_flat_ip_impl
        self.IndexIDMap2: Callable[[FaissIndexProtocol], FaissIndexProtocol] = index_id_map2_impl
        self.normalize_L2: Callable[[VectorArray], None] = normalize_l2_impl

    def index_factory(self, dimension: int, factory_string: str, metric: int) -> FaissIndexProtocol:
        self.index_factory_calls.append((dimension, factory_string, metric))
        return _FakeIndex("factory")

    def write_index(self, index: FaissIndexProtocol, path: str) -> None:
        self.write_calls.append((index, path))

    def read_index(self, path: str) -> FaissIndexProtocol:
        self.read_path = path
        return _FakeIndex("read")


class _ModernFaissModule:
    METRIC_INNER_PRODUCT = 2
    METRIC_L2 = 3

    def __init__(self) -> None:
        self.records: list[tuple[str, object]] = []

    def index_flat_ip(self, dimension: int) -> FaissIndexProtocol:
        self.records.append(("flat", dimension))
        return _FakeIndex("flat")

    def index_factory(self, dimension: int, factory_string: str, metric: int) -> FaissIndexProtocol:
        self.records.append(("factory", (dimension, factory_string, metric)))
        return _FakeIndex("factory")

    def index_id_map2(self, index: FaissIndexProtocol) -> FaissIndexProtocol:
        self.records.append(("idmap", index))
        return _FakeIndex("idmap")

    def write_index(self, index: FaissIndexProtocol, path: str) -> None:
        self.records.append(("write", (index, path)))

    def read_index(self, path: str) -> FaissIndexProtocol:
        self.records.append(("read", path))
        return _FakeIndex("read")

    def normalize_l2(self, vectors: VectorArray) -> None:
        self.records.append(("normalize", vectors.shape))


@pytest.mark.parametrize("dimension", [5, 128])
def test_wrap_faiss_module_adapts_legacy_surface(dimension: int, tmp_path: Path) -> None:
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

    factory_index = adapted.index_factory(dimension, "Flat", adapted.metric_l2)
    assert isinstance(factory_index, _FakeIndex)
    assert legacy.index_factory_calls[-1] == (dimension, "Flat", adapted.metric_l2)

    index_path = tmp_path / "index.faiss"
    adapted.write_index(factory_index, str(index_path))
    assert legacy.write_calls[-1] == (factory_index, str(index_path))

    read_index = adapted.read_index(str(index_path))
    assert isinstance(read_index, _FakeIndex)
    assert legacy.read_path == str(index_path)


def test_wrap_faiss_module_returns_pep8_module_directly() -> None:
    modern = _ModernFaissModule()
    wrapped = wrap_faiss_module(modern)

    # The helper should return the module unchanged when it already satisfies the protocol.
    assert isinstance(wrapped, _ModernFaissModule)
    assert wrapped is modern

    result = wrapped.index_flat_ip(10)
    assert isinstance(result, _FakeIndex)
    assert modern.records[0] == ("flat", 10)
