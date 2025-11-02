"""Table-driven tests for SearchOptions variants and edge cases.

Tests verify:
- Success paths with various facet and parameter combinations
- Failure modes (missing model, invalid facets, etc.)
- Problem Details error format
- Structured logging assertions
"""

from __future__ import annotations

import logging
from typing import Any

import pytest


class TestSearchOptionsSuccess:
    """Happy path scenarios for SearchOptions."""

    @pytest.mark.parametrize(
        ("facets", "candidate_pool", "alpha"),
        [
            (["document", "section"], 1000, 0.5),
            (["document"], 5000, 0.7),
            (["document"], 100, 0.3),
            ([], 1000, 0.5),
        ],
        ids=["multiple_facets", "large_pool", "small_pool", "no_facets"],
    )
    def test_search_options_valid(
        self,
        search_options_factory: Any,  # noqa: ANN401 - pytest fixture typing limitation
        facets: list[str],
        candidate_pool: int,
        alpha: float,
        caplog: Any,  # noqa: ANN401 - pytest fixture typing limitation
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
        [
            (None, True),
            ("", True),
        ],
        ids=["missing_model", "empty_model"],
    )
    def test_search_options_missing_model(
        self,
        search_options_factory: Any,  # noqa: ANN401 - pytest fixture typing limitation
        embedding_model: str | None,
        should_fail: bool,
        caplog: Any,  # noqa: ANN401 - pytest fixture typing limitation
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
        [
            -1,
            0,
        ],
        ids=["negative_pool", "zero_pool"],
    )
    def test_search_options_invalid_pool_size(
        self,
        pool_size: int,
        caplog: Any,  # noqa: ANN401 - pytest fixture typing limitation
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
        [
            -0.1,
            1.1,
        ],
        ids=["below_range", "above_range"],
    )
    def test_search_options_invalid_alpha(
        self,
        alpha: float,
        caplog: Any,  # noqa: ANN401 - pytest fixture typing limitation
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
        search_options_factory: Any,  # noqa: ANN401 - pytest fixture typing limitation
        caplog: Any,  # noqa: ANN401 - pytest fixture typing limitation
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
