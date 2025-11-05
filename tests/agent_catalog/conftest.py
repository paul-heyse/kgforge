"""Fixtures for agent catalog search tests."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, ParamSpec, Protocol, TypedDict, TypeVar, cast

import pytest

# Add src to path for imports
repo_root = Path(__file__).parent.parent.parent
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

P = ParamSpec("P")
R = TypeVar("R")

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

    def fixture(*args: object, **kwargs: object) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Create a pytest fixture.

        Parameters
        ----------
        *args : object
            Positional arguments for pytest.fixture.
        **kwargs : object
            Keyword arguments for pytest.fixture.

        Returns
        -------
        Callable[[Callable[P, R]], Callable[P, R]]
            Decorator function for creating fixtures.
        """
        ...

else:
    fixture = pytest.fixture


class SearchOptionsDict(TypedDict):
    """TypedDict for search options configuration.

    Attributes
    ----------
    embedding_model : str | None
        Embedding model identifier.
    facets : list[str]
        Facet filter keys.
    candidate_pool_size : int
        Number of candidates to consider.
    alpha : float
        Hybrid search weight (0-1).
    """

    embedding_model: str | None
    facets: list[str]
    candidate_pool_size: int
    alpha: float


class SearchOptionsFactory(Protocol):
    """Protocol for search options factory functions."""

    def __call__(
        self,
        *,
        embedding_model: str | None = ...,
        facets: list[str] | None = ...,
        candidate_pool_size: int = ...,
        alpha: float = ...,
    ) -> SearchOptionsDict:
        """Create search options dictionary.

        Parameters
        ----------
        embedding_model : str | None, optional
            Embedding model identifier.
        facets : list[str] | None, optional
            Facet filter keys.
        candidate_pool_size : int, optional
            Number of candidates to consider.
        alpha : float, optional
            Hybrid search weight (0-1).

        Returns
        -------
        SearchOptionsDict
            Search options dictionary.
        """
        ...


@fixture()
def search_options_factory() -> SearchOptionsFactory:
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
    ) -> SearchOptionsDict:
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

    return cast("SearchOptionsFactory", _create_options)


__all__ = ["search_options_factory"]
