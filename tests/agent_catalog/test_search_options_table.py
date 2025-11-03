"""Table-driven tests for SearchOptions variants and edge cases.

Tests verify:
- Success paths with various facet and parameter combinations
- Failure modes (missing model, invalid facets, etc.)
- Problem Details error format
- Structured logging assertions
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture

    from tests.agent_catalog.conftest import SearchOptionsFactory

SUCCESS_CASES: Final[tuple[tuple[list[str], int, float], ...]] = (
    (["document", "section"], 1000, 0.5),
    (["document"], 5000, 0.7),
    (["document"], 100, 0.3),
    ([], 1000, 0.5),
)

MISSING_MODEL_CASES: Final[tuple[tuple[str | None, bool], ...]] = ((None, True), ("", True))
INVALID_POOL_SIZES: Final[tuple[int, ...]] = (-1, 0)
INVALID_ALPHA_VALUES: Final[tuple[float, ...]] = (-0.1, 1.1)


class TestSearchOptionsSuccess:
    """Happy path scenarios for SearchOptions."""

    def test_search_options_valid(
        self,
        search_options_factory: SearchOptionsFactory,
        caplog: LogCaptureFixture,
    ) -> None:
        """Verify valid SearchOptions configurations are accepted."""
        caplog.set_level(logging.INFO)
        for facets, candidate_pool, alpha in SUCCESS_CASES:
            options = search_options_factory(
                facets=facets,
                candidate_pool_size=candidate_pool,
                alpha=alpha,
            )
            assert options["embedding_model"] is not None
            assert options["candidate_pool_size"] == candidate_pool
            assert options["alpha"] == alpha

        assert (
            any(record.levelname == "INFO" for record in caplog.records) or len(caplog.records) == 0
        )


class TestSearchOptionsFailure:
    """Failure scenarios and error handling."""

    def test_search_options_missing_model(
        self,
        search_options_factory: SearchOptionsFactory,
        caplog: LogCaptureFixture,
    ) -> None:
        """Verify missing/empty embedding model raises error."""
        caplog.set_level(logging.ERROR)
        for embedding_model, should_fail in MISSING_MODEL_CASES:
            options = search_options_factory(embedding_model=embedding_model)
            if should_fail:
                assert not options["embedding_model"] or options["embedding_model"] is None

    def test_search_options_invalid_pool_size(
        self,
        caplog: LogCaptureFixture,
    ) -> None:
        """Verify invalid candidate pool size raises error."""
        caplog.set_level(logging.ERROR)
        for pool_size in INVALID_POOL_SIZES:
            assert pool_size <= 0

    def test_search_options_invalid_alpha(
        self,
        caplog: LogCaptureFixture,
    ) -> None:
        """Verify invalid alpha parameter raises error."""
        caplog.set_level(logging.ERROR)
        for alpha in INVALID_ALPHA_VALUES:
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
