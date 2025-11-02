"""Fixtures for agent catalog search tests."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

# Add src to path for imports
repo_root = Path(__file__).parent.parent.parent
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


@pytest.fixture
def search_options_factory() -> Any:  # noqa: ANN401 - pytest fixture typing limitation
    """Create a factory for SearchOptions test cases.

    Yields
    ------
    callable
        Function to create SearchOptions with specified parameters.
    """

    def _create_options(
        embedding_model: str | None = "default-model",
        facets: list[str] | None = None,
        candidate_pool_size: int = 1000,
        alpha: float = 0.5,
    ) -> dict[str, Any]:
        """Create SearchOptions dict for testing.

        Parameters
        ----------
        embedding_model : str | None, optional
            Embedding model name.
        facets : list[str] | None, optional
            Filter facets.
        candidate_pool_size : int, optional
            Candidate pool size.
        alpha : float, optional
            Weighting factor.

        Returns
        -------
        dict
            SearchOptions configuration dict.
        """
        return {
            "embedding_model": embedding_model,
            "facets": facets or ["default"],
            "candidate_pool_size": candidate_pool_size,
            "alpha": alpha,
        }

    return _create_options


__all__ = ["search_options_factory"]
