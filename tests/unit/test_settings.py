"""Unit tests for RuntimeSettings and configuration models.

Tests verify fail-fast validation, environment variable loading,
nested model support, and Problem Details error reporting.
"""

from __future__ import annotations

import pytest

from kgfoundry_common.errors import ConfigurationError
from kgfoundry_common.settings import (
    FaissConfig,
    ObservabilityConfig,
    RuntimeSettings,
    SearchConfig,
    SparseEmbeddingConfig,
)


class TestSearchConfig:
    """Test SearchConfig nested model."""

    def test_search_config_defaults(self) -> None:
        """SearchConfig has sensible defaults."""
        config = SearchConfig()
        assert config.k == 10
        assert config.dense_candidates == 200
        assert config.sparse_candidates == 200
        assert config.api_url is None

    def test_search_config_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SearchConfig loads from environment variables."""
        monkeypatch.setenv("KGFOUNDRY_SEARCH_API_URL", "http://localhost:8000")
        monkeypatch.setenv("KGFOUNDRY_SEARCH_K", "20")
        config = SearchConfig()
        assert config.api_url == "http://localhost:8000"
        assert config.k == 20


class TestObservabilityConfig:
    """Test ObservabilityConfig nested model."""

    def test_observability_config_defaults(self) -> None:
        """ObservabilityConfig has sensible defaults."""
        config = ObservabilityConfig()
        assert config.log_level == "INFO"
        assert config.metrics_enabled is True
        assert config.traces_enabled is False
        assert config.metrics_port == 9090


class TestSparseEmbeddingConfig:
    """Test SparseEmbeddingConfig nested model."""

    def test_sparse_embedding_config_defaults(self) -> None:
        """SparseEmbeddingConfig has sensible defaults."""
        config = SparseEmbeddingConfig()
        assert config.bm25_index_dir == "./_indices/bm25"
        assert config.bm25_k1 == 0.9
        assert config.bm25_b == 0.4
        assert config.splade_index_dir == "./_indices/splade_impact"
        assert config.splade_query_encoder == "naver/splade-v3-distilbert"

    def test_sparse_embedding_config_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SparseEmbeddingConfig loads from environment variables."""
        monkeypatch.setenv("KGFOUNDRY_SPARSE_EMBEDDING_BM25_INDEX_DIR", "/custom/bm25")
        monkeypatch.setenv("KGFOUNDRY_SPARSE_EMBEDDING_BM25_K1", "1.2")
        config = SparseEmbeddingConfig()
        assert config.bm25_index_dir == "/custom/bm25"
        assert config.bm25_k1 == 1.2


class TestFaissConfig:
    """Test FaissConfig nested model."""

    def test_faiss_config_defaults(self) -> None:
        """FaissConfig has sensible defaults."""
        config = FaissConfig()
        assert config.gpu is True
        assert config.cuvs is True
        assert config.index_factory == "OPQ64,IVF8192,PQ64"
        assert config.nprobe == 64
        assert config.index_path == "./_indices/faiss/shard_000.idx"

    def test_faiss_config_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """FaissConfig loads from environment variables."""
        monkeypatch.setenv("KGFOUNDRY_FAISS_GPU", "false")
        monkeypatch.setenv("KGFOUNDRY_FAISS_NPROBE", "128")
        config = FaissConfig()
        assert config.gpu is False
        assert config.nprobe == 128


class TestRuntimeSettings:
    """Test RuntimeSettings with nested models."""

    def test_runtime_settings_defaults(self) -> None:
        """RuntimeSettings loads with default values."""
        settings = RuntimeSettings()
        assert settings.search.k == 10
        assert settings.search.sparse_backend == "lucene"
        assert settings.observability.log_level == "INFO"
        assert settings.sparse_embedding.bm25_index_dir == "./_indices/bm25"
        assert settings.faiss.gpu is True

    def test_observability_config_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ObservabilityConfig loads from environment variables."""
        monkeypatch.setenv("KGFOUNDRY_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("KGFOUNDRY_METRICS_ENABLED", "false")
        config = ObservabilityConfig()
        assert config.log_level == "DEBUG"
        assert config.metrics_enabled is False


class TestRuntimeSettings:
    """Test RuntimeSettings fail-fast validation."""

    def test_runtime_settings_defaults(self) -> None:
        """RuntimeSettings initializes with defaults."""
        settings = RuntimeSettings()
        assert settings.search.k == 10
        assert settings.observability.log_level == "INFO"

    def test_runtime_settings_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RuntimeSettings loads nested config from environment variables."""
        monkeypatch.setenv("KGFOUNDRY_SEARCH__K", "25")
        monkeypatch.setenv("KGFOUNDRY_OBSERVABILITY__LOG_LEVEL", "DEBUG")
        settings = RuntimeSettings()
        assert settings.search.k == 25
        assert settings.observability.log_level == "DEBUG"

    def test_runtime_settings_validation_error_raises_configuration_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invalid settings raise ConfigurationError with Problem Details."""
        monkeypatch.setenv("KGFOUNDRY_SEARCH__K", "not-a-number")
        with pytest.raises(ConfigurationError) as exc_info:
            RuntimeSettings()

        error = exc_info.value
        assert error.code.value == "configuration-error"
        assert "validation_error" in error.context

    def test_runtime_settings_extra_fields_forbidden(self) -> None:
        """RuntimeSettings rejects unknown fields (extra='forbid')."""
        with pytest.raises(ConfigurationError, match="Extra inputs are not permitted"):
            RuntimeSettings(unknown_field="value")  # type: ignore[call-arg]
