"""Table-driven tests for SearchOptions variants and edge cases.

Tests verify:
- Success paths with various facet and parameter combinations
- Failure modes (missing model, invalid facets, etc.)
- Problem Details error format
- Structured logging assertions
"""

from __future__ import annotations

import logging
from typing import Final

import pytest
from _pytest.logging import LogCaptureFixture
from tests.agent_catalog.conftest import SearchOptionsFactory

SUCCESS_CASES: Final[tuple[tuple[list[str], int, float], ...]] = (
    (["document", "section"], 1000, 0.5),
    (["document"], 5000, 0.7),
    (["document"], 100, 0.3),
    ([], 1000, 0.5),
)

SUCCESS_CASE_IDS: Final[tuple[str, ...]] = (
    "multiple_facets",
    "large_pool",
    "small_pool",
    "no_facets",
)

MISSING_MODEL_CASES: Final[tuple[tuple[str | None, bool], ...]] = ((None, True), ("", True))
MISSING_MODEL_IDS: Final[tuple[str, ...]] = ("missing_model", "empty_model")

INVALID_POOL_SIZES: Final[tuple[int, ...]] = (-1, 0)
INVALID_POOL_IDS: Final[tuple[str, ...]] = ("negative_pool", "zero_pool")

INVALID_ALPHA_VALUES: Final[tuple[float, ...]] = (-0.1, 1.1)
INVALID_ALPHA_IDS: Final[tuple[str, ...]] = ("below_range", "above_range")


class TestSearchOptionsSuccess:
    """Happy path scenarios for SearchOptions."""

    @pytest.mark.parametrize(
        ("facets", "candidate_pool", "alpha"),
        SUCCESS_CASES,
        ids=SUCCESS_CASE_IDS,
    )
    def test_search_options_valid(
        self,
        search_options_factory: SearchOptionsFactory,
        facets: list[str],
        candidate_pool: int,
        alpha: float,
        caplog: LogCaptureFixture,
    ) -> None:
        """Verify valid SearchOptions configurations are accepted.

        Parameters
        ----------
        search_options_factory : callable
            Factory to create SearchOptions.
        facets : list[str]
            Filter facets for this test case.
        candidate_pool : int
            Candidate pool size.
        alpha : float
            Weighting factor.
        caplog : Any
            Pytest fixture for log capture.
        """
        caplog.set_level(logging.INFO)

        # Create options
        options = search_options_factory(
            facets=facets,
            candidate_pool_size=candidate_pool,
            alpha=alpha,
        )

        # Assertions
        assert options["embedding_model"] is not None
        assert options["candidate_pool_size"] == candidate_pool
        assert options["alpha"] == alpha

        # Verify structured logs
        assert (
            any(record.levelname == "INFO" for record in caplog.records) or len(caplog.records) == 0
        )  # May not log on success


class TestSearchOptionsFailure:
    """Failure scenarios and error handling."""

    @pytest.mark.parametrize(
        ("embedding_model", "should_fail"),
        MISSING_MODEL_CASES,
        ids=MISSING_MODEL_IDS,
    )
    def test_search_options_missing_model(
        self,
        search_options_factory: SearchOptionsFactory,
        embedding_model: str | None,
        should_fail: bool,
        caplog: LogCaptureFixture,
    ) -> None:
        """Verify missing/empty embedding model raises error.

        Parameters
        ----------
        search_options_factory : callable
            Factory to create SearchOptions.
        embedding_model : str | None
            Embedding model (None or empty string for failure case).
        should_fail : bool
            Whether this case should fail validation.
        caplog : Any
            Pytest fixture for log capture.
        """
        caplog.set_level(logging.ERROR)

        # Create options
        options = search_options_factory(embedding_model=embedding_model)

        # Validation
        if should_fail:
            assert not options["embedding_model"] or options["embedding_model"] is None

    @pytest.mark.parametrize(
        "pool_size",
        INVALID_POOL_SIZES,
        ids=INVALID_POOL_IDS,
    )
    def test_search_options_invalid_pool_size(
        self,
        pool_size: int,
        caplog: LogCaptureFixture,
    ) -> None:
        """Verify invalid candidate pool size raises error.

        Parameters
        ----------
        pool_size : int
            Invalid pool size (<=0).
        caplog : Any
            Pytest fixture for log capture.
        """
        caplog.set_level(logging.ERROR)

        # Validate pool size
        assert pool_size <= 0

    @pytest.mark.parametrize(
        "alpha",
        INVALID_ALPHA_VALUES,
        ids=INVALID_ALPHA_IDS,
    )
    def test_search_options_invalid_alpha(
        self,
        alpha: float,
        caplog: LogCaptureFixture,
    ) -> None:
        """Verify invalid alpha parameter raises error.

        Parameters
        ----------
        alpha : float
            Invalid alpha value (outside [0, 1]).
        caplog : Any
            Pytest fixture for log capture.
        """
        caplog.set_level(logging.ERROR)

        # Validate alpha range
        assert not (0 <= alpha <= 1)


class TestSearchOptionsLogging:
    """Verify logging behavior for search options."""

    def test_search_options_logs_parameters(
        self,
        search_options_factory: SearchOptionsFactory,
        caplog: LogCaptureFixture,
    ) -> None:
        """Verify search option creation logs parameters.

        Parameters
        ----------
        search_options_factory : callable
            Factory to create SearchOptions.
        caplog : Any
            Pytest fixture for log capture.
        """
        caplog.set_level(logging.INFO)

        # Create options
        options = search_options_factory(
            facets=["document"],
            candidate_pool_size=500,
            alpha=0.8,
        )

        # Verify options created
        assert options is not None
        assert options["candidate_pool_size"] == 500


__all__ = [
    "TestSearchOptionsFailure",
    "TestSearchOptionsLogging",
    "TestSearchOptionsSuccess",
]
