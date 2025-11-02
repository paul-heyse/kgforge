"""Tests for SearchOptions helper factories and validation.

Tests are organized in classes for logical grouping and test isolation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pytest

from kgfoundry.agent_catalog.search import (
    EmbeddingModelProtocol,
    SearchOptions,
    build_default_search_options,
    build_embedding_aware_search_options,
    build_faceted_search_options,
    make_search_document,
)
from kgfoundry_common.errors import AgentCatalogSearchError
from search_api.types import VectorArray


@pytest.fixture
def mock_embedding_loader() -> Callable[[str], EmbeddingModelProtocol]:
    """Fixture providing a mock embedding model loader."""

    class MockEmbeddingModel(EmbeddingModelProtocol):
        """Mock embedding model implementing EmbeddingModelProtocol."""

        def encode(self, sentences: Sequence[str], **_kwargs: object) -> VectorArray:
            """Mock encode method returning dummy embeddings."""
            if not sentences:
                return np.zeros((0, 10), dtype=np.float32)
            return np.full((len(sentences), 10), 0.1, dtype=np.float32)

    def loader(_name: str) -> MockEmbeddingModel:
        return MockEmbeddingModel()

    return loader


class TestBuildDefaultSearchOptions:
    """Tests for build_default_search_options helper factory."""

    def test_default_values_applied(self) -> None:
        """Test that canonical defaults are applied when params omitted."""
        opts = build_default_search_options()
        assert opts.alpha == 0.6
        assert opts.candidate_pool == 100
        assert opts.batch_size == 32
        assert opts.facets is None
        assert opts.embedding_model is None
        assert opts.model_loader is None

    def test_custom_alpha(self) -> None:
        """Test that custom alpha value is applied."""
        opts = build_default_search_options(alpha=0.3)
        assert opts.alpha == 0.3
        assert opts.candidate_pool == 100  # defaults applied

    def test_custom_candidate_pool(self) -> None:
        """Test that custom candidate pool is applied."""
        opts = build_default_search_options(candidate_pool=50)
        assert opts.candidate_pool == 50
        assert opts.alpha == 0.6

    @pytest.mark.parametrize(
        "alpha",
        [-0.1, 1.1, -1.0, 2.0],
    )
    def test_alpha_validation_rejects_out_of_range(self, alpha: float) -> None:
        """Test that alpha outside [0.0, 1.0] raises AgentCatalogSearchError."""
        with pytest.raises(AgentCatalogSearchError) as exc_info:
            build_default_search_options(alpha=alpha)
        assert "alpha must be in [0.0, 1.0]" in str(exc_info.value)

    @pytest.mark.parametrize(
        "alpha",
        [0.0, 0.5, 1.0],
    )
    def test_alpha_validation_accepts_in_range(self, alpha: float) -> None:
        """Test that alpha in [0.0, 1.0] is accepted."""
        opts = build_default_search_options(alpha=alpha)
        assert opts.alpha == alpha

    @pytest.mark.parametrize(
        "pool",
        [-1, -10],
    )
    def test_candidate_pool_validation_rejects_negative(self, pool: int) -> None:
        """Test that negative candidate pool raises AgentCatalogSearchError."""
        with pytest.raises(AgentCatalogSearchError) as exc_info:
            build_default_search_options(candidate_pool=pool)
        assert "candidate_pool must be non-negative" in str(exc_info.value)

    def test_all_parameters(
        self, mock_embedding_loader: Callable[[str], EmbeddingModelProtocol]
    ) -> None:
        """Test that all parameters can be set."""
        opts = build_default_search_options(
            alpha=0.7,
            candidate_pool=200,
            batch_size=64,
            embedding_model="test-model",
            model_loader=mock_embedding_loader,
        )
        assert opts.alpha == 0.7
        assert opts.candidate_pool == 200
        assert opts.batch_size == 64
        assert opts.embedding_model == "test-model"
        assert opts.model_loader is mock_embedding_loader


class TestBuildFacetedSearchOptions:
    """Tests for build_faceted_search_options helper factory."""

    def test_facets_included(self) -> None:
        """Test that facets are properly included in options."""
        facets = {"package": "kgfoundry", "kind": "class"}
        opts = build_faceted_search_options(facets=facets)
        assert opts.facets == facets
        assert opts.alpha == 0.6  # defaults applied

    def test_empty_facets_allowed(self) -> None:
        """Test that empty facets dict is allowed."""
        opts = build_faceted_search_options(facets={})
        assert opts.facets == {}

    @pytest.mark.parametrize(
        "invalid_facets",
        [
            {"invalid_key": "value"},
            {"package": "pkg", "bad_key": "val"},
            {"unknown": "facet"},
        ],
    )
    def test_invalid_facet_keys_rejected(self, invalid_facets: dict[str, str]) -> None:
        """Test that unrecognized facet keys are rejected."""
        with pytest.raises(AgentCatalogSearchError) as exc_info:
            build_faceted_search_options(facets=invalid_facets)
        assert "Invalid facet keys" in str(exc_info.value)

    def test_allowed_facet_keys(self) -> None:
        """Test that all allowed facet keys are accepted."""
        allowed = {
            "package": "pkg",
            "module": "mod",
            "kind": "function",
            "stability": "stable",
        }
        opts = build_faceted_search_options(facets=allowed)
        assert opts.facets == allowed

    def test_custom_params_with_facets(self) -> None:
        """Test that custom params can be combined with facets."""
        facets = {"package": "test"}
        opts = build_faceted_search_options(
            facets=facets,
            alpha=0.4,
            candidate_pool=50,
        )
        assert opts.facets == facets
        assert opts.alpha == 0.4
        assert opts.candidate_pool == 50


class TestBuildEmbeddingAwareSearchOptions:
    """Tests for build_embedding_aware_search_options helper factory."""

    def test_embedding_model_and_loader_required(
        self, mock_embedding_loader: Callable[[str], EmbeddingModelProtocol]
    ) -> None:
        """Test that embedding_model and model_loader are required."""
        opts = build_embedding_aware_search_options(
            embedding_model="all-MiniLM-L6-v2",
            model_loader=mock_embedding_loader,
        )
        assert opts.embedding_model == "all-MiniLM-L6-v2"
        assert opts.model_loader is mock_embedding_loader
        assert opts.alpha == 0.6  # defaults

    def test_optional_facets_with_embedding(
        self, mock_embedding_loader: Callable[[str], EmbeddingModelProtocol]
    ) -> None:
        """Test that facets can be combined with embedding options."""
        facets = {"kind": "class"}
        opts = build_embedding_aware_search_options(
            embedding_model="model",
            model_loader=mock_embedding_loader,
            facets=facets,
        )
        assert opts.facets == facets
        assert opts.embedding_model == "model"

    def test_invalid_facets_with_embedding_rejected(
        self, mock_embedding_loader: Callable[[str], EmbeddingModelProtocol]
    ) -> None:
        """Test that invalid facets are rejected even with embedding."""
        with pytest.raises(AgentCatalogSearchError) as exc_info:
            build_embedding_aware_search_options(
                embedding_model="model",
                model_loader=mock_embedding_loader,
                facets={"bad_key": "val"},
            )
        assert "Invalid facet keys" in str(exc_info.value)


class TestMakeSearchDocument:
    """Tests for make_search_document helper."""

    def test_minimal_document(self) -> None:
        """Test creation of minimal valid SearchDocument."""
        doc = make_search_document(
            symbol_id="py:test.func",
            package="test",
            module="test.module",
            qname="func",
            kind="function",
        )
        assert doc.symbol_id == "py:test.func"
        assert doc.package == "test"
        assert doc.module == "test.module"
        assert doc.qname == "func"
        assert doc.kind == "function"
        assert doc.stability is None
        assert doc.deprecated is False
        assert doc.row == -1

    def test_whitespace_normalization(self) -> None:
        """Test that whitespace is stripped from text fields."""
        doc = make_search_document(
            symbol_id="py:test",
            package=" package ",
            module=" module ",
            qname=" qname ",
            kind="class",
            summary="  summary  ",
            docstring="  docstring  ",
        )
        assert doc.package == " package "  # dataclass field preserved
        assert doc.summary == "summary"  # normalized in helper
        assert doc.docstring == "docstring"  # normalized in helper
        assert doc.qname == "qname"  # normalized in helper

    def test_text_field_construction(self) -> None:
        """Test that text field is constructed from normalized parts."""
        doc = make_search_document(
            symbol_id="py:test",
            package="pkg",
            module="mod",
            qname="name",
            kind="function",
            summary="A function",
            docstring="Does something",
        )
        # Text should contain all parts joined
        assert "name" in doc.text
        assert "mod" in doc.text
        assert "pkg" in doc.text
        assert "A function" in doc.text
        assert "Does something" in doc.text

    def test_tokens_generated(self) -> None:
        """Test that lexical tokens are generated from text."""
        doc = make_search_document(
            symbol_id="py:test",
            package="kgfoundry",
            module="agent_catalog.search",
            qname="find_similar",
            kind="function",
            summary="Find similar symbols",
        )
        # Should have tokens from all fields
        assert len(doc.tokens) > 0
        # Check some key tokens exist
        assert "find" in doc.tokens or "similar" in doc.tokens

    def test_full_document_with_anchors(self) -> None:
        """Test creation of document with all fields including anchors."""
        doc = make_search_document(
            symbol_id="py:kgfoundry.search.find",
            package="kgfoundry",
            module="kgfoundry.agent_catalog.search",
            qname="find",
            kind="function",
            stability="stable",
            deprecated=True,
            summary="Find symbols",
            docstring="Detailed docs",
            anchor_start=100,
            anchor_end=150,
            row=5,
        )
        assert doc.stability == "stable"
        assert doc.deprecated is True
        assert doc.anchor_start == 100
        assert doc.anchor_end == 150
        assert doc.row == 5


class TestSearchOptionsRoundTrip:
    """Tests for round-trip consistency of SearchOptions."""

    def test_options_are_valid_after_build(
        self, mock_embedding_loader: Callable[[str], EmbeddingModelProtocol]
    ) -> None:
        """Test that built options are valid SearchOptions instances."""
        opts1 = build_default_search_options(alpha=0.7)
        opts2 = build_faceted_search_options(facets={"package": "test"})

        opts3 = build_embedding_aware_search_options(
            embedding_model="model",
            model_loader=mock_embedding_loader,
        )

        # All should be SearchOptions instances
        assert isinstance(opts1, SearchOptions)
        assert isinstance(opts2, SearchOptions)
        assert isinstance(opts3, SearchOptions)

    def test_helper_consistency(self) -> None:
        """Test that different helpers produce consistent option objects."""
        opts1 = build_default_search_options(alpha=0.5, candidate_pool=75)
        opts2 = SearchOptions(alpha=0.5, candidate_pool=75)

        assert opts1.alpha == opts2.alpha
        assert opts1.candidate_pool == opts2.candidate_pool
