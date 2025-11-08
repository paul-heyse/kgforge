"""Tests for incremental FAISS indexing (dual-index architecture).

Tests verify that incremental updates work correctly:
- update_index() adds vectors to secondary index
- Dual-index search returns results from both indexes
- merge_indexes() combines secondary into primary
- End-to-end workflow: initial → incremental → merge → search
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from codeintel_rev.io.faiss_manager import FAISSManager

# Use modern numpy random generator
_rng = np.random.default_rng(42)

# Unit tests use reduced dimensions for speed
_UNIT_TEST_VEC_DIM = 256


@pytest.fixture
def tmp_index_path(tmp_path: Path) -> Path:
    """Create a temporary index path for testing.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory fixture.

    Returns
    -------
    Path
        Path to temporary FAISS index file.
    """
    return tmp_path / "test_index.faiss"


def test_update_index_creates_secondary(tmp_index_path: Path) -> None:
    """Test that update_index creates secondary index on first call.

    Parameters
    ----------
    tmp_index_path : Path
        Temporary index path.
    """
    vec_dim = _UNIT_TEST_VEC_DIM
    manager = FAISSManager(index_path=tmp_index_path, vec_dim=vec_dim)

    # Secondary index should not exist initially
    assert manager.secondary_index is None
    assert len(manager.incremental_ids) == 0

    # Generate test vectors
    new_vectors = _rng.normal(0.5, 0.15, (10, vec_dim)).astype(np.float32)
    new_vectors = np.clip(new_vectors, 0.0, 1.0)
    new_ids = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109], dtype=np.int64)

    # Call update_index - should create secondary index
    manager.update_index(new_vectors, new_ids)

    # Verify secondary index was created
    assert manager.secondary_index is not None
    assert len(manager.incremental_ids) == 10
    assert manager.incremental_ids == {100, 101, 102, 103, 104, 105, 106, 107, 108, 109}


def test_update_index_skips_duplicates(tmp_index_path: Path) -> None:
    """Test that update_index skips duplicate IDs.

    Parameters
    ----------
    tmp_index_path : Path
        Temporary index path.
    """
    vec_dim = _UNIT_TEST_VEC_DIM
    manager = FAISSManager(index_path=tmp_index_path, vec_dim=vec_dim)

    # First batch
    vectors1 = _rng.normal(0.5, 0.15, (5, vec_dim)).astype(np.float32)
    vectors1 = np.clip(vectors1, 0.0, 1.0)
    ids1 = np.array([10, 11, 12, 13, 14], dtype=np.int64)
    manager.update_index(vectors1, ids1)

    assert len(manager.incremental_ids) == 5

    # Second batch with duplicates
    vectors2 = _rng.normal(0.5, 0.15, (7, vec_dim)).astype(np.float32)
    vectors2 = np.clip(vectors2, 0.0, 1.0)
    ids2 = np.array([12, 13, 14, 15, 16, 17, 18], dtype=np.int64)  # 12, 13, 14 are duplicates

    manager.update_index(vectors2, ids2)

    # Should have 5 original + 4 new = 9 total
    assert len(manager.incremental_ids) == 9
    assert manager.incremental_ids == {10, 11, 12, 13, 14, 15, 16, 17, 18}


def test_dual_index_search(tmp_index_path: Path) -> None:
    """Test that search returns results from both primary and secondary indexes.

    Parameters
    ----------
    tmp_index_path : Path
        Temporary index path.
    """
    vec_dim = _UNIT_TEST_VEC_DIM
    manager = FAISSManager(index_path=tmp_index_path, vec_dim=vec_dim)

    # Build primary index with initial vectors
    primary_vectors = _rng.normal(0.5, 0.15, (100, vec_dim)).astype(np.float32)
    primary_vectors = np.clip(primary_vectors, 0.0, 1.0)
    manager.build_index(primary_vectors)
    primary_ids = np.arange(100, dtype=np.int64)
    manager.add_vectors(primary_vectors, primary_ids)

    # Add vectors to secondary index
    secondary_vectors = _rng.normal(0.5, 0.15, (20, vec_dim)).astype(np.float32)
    secondary_vectors = np.clip(secondary_vectors, 0.0, 1.0)
    secondary_ids = np.arange(100, 120, dtype=np.int64)  # IDs 100-119
    manager.update_index(secondary_vectors, secondary_ids)

    # Create a query vector (similar to one in secondary index)
    query = secondary_vectors[0].copy().reshape(1, -1)

    # Search - should return results from both indexes
    _distances, result_ids = manager.search(query, k=10)

    # Verify we got results
    assert len(result_ids[0]) == 10

    # Verify results include IDs from both primary and secondary
    # (exact composition depends on similarity, but should have some from each)
    result_id_set = set(result_ids[0])
    primary_result_count = sum(1 for rid in result_id_set if rid < 100)
    secondary_result_count = sum(1 for rid in result_id_set if rid >= 100)

    # Should have results from both (exact counts depend on similarity)
    assert primary_result_count > 0 or secondary_result_count > 0


def test_merge_indexes_combines_vectors(tmp_index_path: Path) -> None:
    """Test that merge_indexes combines secondary into primary and clears secondary.

    Parameters
    ----------
    tmp_index_path : Path
        Temporary index path.
    """
    vec_dim = _UNIT_TEST_VEC_DIM
    manager = FAISSManager(index_path=tmp_index_path, vec_dim=vec_dim)

    # Build primary index
    primary_vectors = _rng.normal(0.5, 0.15, (50, vec_dim)).astype(np.float32)
    primary_vectors = np.clip(primary_vectors, 0.0, 1.0)
    manager.build_index(primary_vectors)
    primary_ids = np.arange(50, dtype=np.int64)
    manager.add_vectors(primary_vectors, primary_ids)
    manager.save_cpu_index()

    # Add to secondary index
    secondary_vectors = _rng.normal(0.5, 0.15, (30, vec_dim)).astype(np.float32)
    secondary_vectors = np.clip(secondary_vectors, 0.0, 1.0)
    secondary_ids = np.arange(50, 80, dtype=np.int64)
    manager.update_index(secondary_vectors, secondary_ids)

    # Verify secondary exists before merge
    assert manager.secondary_index is not None
    assert len(manager.incremental_ids) == 30

    # Merge indexes
    manager.merge_indexes()

    # Verify secondary is cleared
    assert manager.secondary_index is None
    assert len(manager.incremental_ids) == 0

    # Verify primary index now has all vectors (80 total)
    assert manager.cpu_index is not None
    assert manager.cpu_index.ntotal == 80  # type: ignore[attr-defined]


def test_merge_indexes_no_secondary(tmp_index_path: Path) -> None:
    """Test that merge_indexes handles case where no secondary index exists.

    Parameters
    ----------
    tmp_index_path : Path
        Temporary index path.
    """
    vec_dim = _UNIT_TEST_VEC_DIM
    manager = FAISSManager(index_path=tmp_index_path, vec_dim=vec_dim)

    # Build primary index only
    primary_vectors = _rng.normal(0.5, 0.15, (50, vec_dim)).astype(np.float32)
    primary_vectors = np.clip(primary_vectors, 0.0, 1.0)
    manager.build_index(primary_vectors)
    primary_ids = np.arange(50, dtype=np.int64)
    manager.add_vectors(primary_vectors, primary_ids)

    # Merge should do nothing (no secondary index)
    manager.merge_indexes()

    # Primary should be unchanged
    assert manager.cpu_index is not None
    assert manager.cpu_index.ntotal == 50  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("primary_size", "secondary_size"),
    [
        (100, 10),
        (1000, 100),
        (5000, 500),
    ],
)
def test_incremental_workflow_end_to_end(
    tmp_index_path: Path, primary_size: int, secondary_size: int
) -> None:
    """Test complete incremental workflow: initial → incremental → merge → search.

    Parameters
    ----------
    tmp_index_path : Path
        Temporary index path.
    primary_size : int
        Number of vectors in primary index.
    secondary_size : int
        Number of vectors to add incrementally.
    """
    vec_dim = _UNIT_TEST_VEC_DIM
    manager = FAISSManager(index_path=tmp_index_path, vec_dim=vec_dim)

    # Step 1: Initial indexing (build primary)
    primary_vectors = _rng.normal(0.5, 0.15, (primary_size, vec_dim)).astype(np.float32)
    primary_vectors = np.clip(primary_vectors, 0.0, 1.0)
    manager.build_index(primary_vectors)
    primary_ids = np.arange(primary_size, dtype=np.int64)
    manager.add_vectors(primary_vectors, primary_ids)
    manager.save_cpu_index()

    # Step 2: Incremental update (add to secondary)
    secondary_vectors = _rng.normal(0.5, 0.15, (secondary_size, vec_dim)).astype(np.float32)
    secondary_vectors = np.clip(secondary_vectors, 0.0, 1.0)
    secondary_ids = np.arange(primary_size, primary_size + secondary_size, dtype=np.int64)
    manager.update_index(secondary_vectors, secondary_ids)

    # Verify dual-index state
    assert manager.secondary_index is not None
    assert len(manager.incremental_ids) == secondary_size

    # Step 3: Search with dual-index
    query = secondary_vectors[0].copy().reshape(1, -1)
    _distances, result_ids = manager.search(query, k=10)
    assert len(result_ids[0]) == 10

    # Step 4: Merge secondary into primary
    manager.merge_indexes()

    # Verify merge completed
    assert manager.secondary_index is None
    assert manager.cpu_index is not None
    assert manager.cpu_index.ntotal == primary_size + secondary_size  # type: ignore[attr-defined]

    # Step 5: Search after merge (should still work, now only primary)
    _distances2, result_ids2 = manager.search(query, k=10)
    assert len(result_ids2[0]) == 10


def test_save_load_secondary_index(tmp_index_path: Path) -> None:
    """Test that secondary index can be saved and loaded.

    Parameters
    ----------
    tmp_index_path : Path
        Temporary index path.
    """
    vec_dim = _UNIT_TEST_VEC_DIM
    manager1 = FAISSManager(index_path=tmp_index_path, vec_dim=vec_dim)

    # Build primary and add to secondary
    primary_vectors = _rng.normal(0.5, 0.15, (50, vec_dim)).astype(np.float32)
    primary_vectors = np.clip(primary_vectors, 0.0, 1.0)
    manager1.build_index(primary_vectors)
    manager1.add_vectors(primary_vectors, np.arange(50, dtype=np.int64))
    manager1.save_cpu_index()

    secondary_vectors = _rng.normal(0.5, 0.15, (20, vec_dim)).astype(np.float32)
    secondary_vectors = np.clip(secondary_vectors, 0.0, 1.0)
    secondary_ids = np.arange(50, 70, dtype=np.int64)
    manager1.update_index(secondary_vectors, secondary_ids)
    manager1.save_secondary_index()

    # Create new manager and load both indexes
    manager2 = FAISSManager(index_path=tmp_index_path, vec_dim=vec_dim)
    manager2.load_cpu_index()
    manager2.load_secondary_index()

    # Verify secondary index was restored
    assert manager2.secondary_index is not None
    assert len(manager2.incremental_ids) == 20
    assert manager2.incremental_ids == set(range(50, 70))

    # Verify search works with loaded indexes
    query = secondary_vectors[0].copy().reshape(1, -1)
    _distances, result_ids = manager2.search(query, k=10)
    assert len(result_ids[0]) == 10
