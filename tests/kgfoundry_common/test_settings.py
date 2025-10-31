"""Tests for kgfoundry_common.settings module.

Tests cover settings loading, environment variable overrides, missing env failures,
and Problem Details conversion.
"""

from __future__ import annotations

import os

import pytest

from kgfoundry_common.errors import SettingsError
from kgfoundry_common.problem_details import ProblemDetails
from kgfoundry_common.settings import (
    FaissConfig,
    KgFoundrySettings,
    ObservabilityConfig,
    RuntimeSettings,
    SearchConfig,
    SparseEmbeddingConfig,
    load_settings,
)


class TestSearchConfig:
    """Tests for SearchConfig."""

    def test_defaults(self) -> None:
        """SearchConfig uses correct defaults."""
        config = SearchConfig()
        assert config.api_url is None
        assert config.k == 10
        assert config.dense_candidates == 200
        assert config.sparse_candidates == 200
        assert config.rrf_k == 60
        assert config.sparse_backend == "lucene"
        assert config.kg_boosts_direct == 0.08
        assert config.kg_boosts_one_hop == 0.04

    def test_env_override(self) -> None:
        """SearchConfig loads from environment variables."""
        os.environ["KGFOUNDRY_SEARCH_API_URL"] = "http://test:8000"
        os.environ["KGFOUNDRY_SEARCH_K"] = "20"
        try:
            config = SearchConfig()
            assert config.api_url == "http://test:8000"
            assert config.k == 20
        finally:
            os.environ.pop("KGFOUNDRY_SEARCH_API_URL", None)
            os.environ.pop("KGFOUNDRY_SEARCH_K", None)

    def test_extra_fields_forbidden(self) -> None:
        """SearchConfig rejects unknown fields."""
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):  # type: ignore[call-arg]  # pytest.raises supports match, mypy stub issue
            SearchConfig(unknown_field="value")  # type: ignore[call-arg]  # Intentionally invalid


class TestObservabilityConfig:
    """Tests for ObservabilityConfig."""

    def test_defaults(self) -> None:
        """ObservabilityConfig uses correct defaults."""
        config = ObservabilityConfig()
        assert config.log_level == "INFO"
        assert config.metrics_enabled is True
        assert config.traces_enabled is False
        assert config.metrics_port == 9090


class TestRuntimeSettings:
    """Tests for RuntimeSettings."""

    def test_defaults(self) -> None:
        """RuntimeSettings uses correct defaults."""
        settings = RuntimeSettings()
        assert isinstance(settings.search, SearchConfig)  # type: ignore[misc]  # pydantic models use Any internally
        assert isinstance(settings.observability, ObservabilityConfig)  # type: ignore[misc]  # pydantic models use Any internally
        assert isinstance(settings.sparse_embedding, SparseEmbeddingConfig)  # type: ignore[misc]  # pydantic models use Any internally
        assert isinstance(settings.faiss, FaissConfig)  # type: ignore[misc]  # pydantic models use Any internally

    def test_env_override(self) -> None:
        """RuntimeSettings loads from environment variables."""
        os.environ["KGFOUNDRY_SEARCH__API_URL"] = "http://test:8000"
        os.environ["KGFOUNDRY_OBSERVABILITY__LOG_LEVEL"] = "DEBUG"
        try:
            settings = RuntimeSettings()
            assert settings.search.api_url == "http://test:8000"
            assert settings.observability.log_level == "DEBUG"
        finally:
            os.environ.pop("KGFOUNDRY_SEARCH__API_URL", None)
            os.environ.pop("KGFOUNDRY_OBSERVABILITY__LOG_LEVEL", None)

    def test_programmatic_override(self) -> None:
        """RuntimeSettings accepts programmatic overrides."""
        settings = RuntimeSettings(
            search={"api_url": "http://override:8000"},
            observability={"log_level": "WARNING"},
        )
        assert settings.search.api_url == "http://override:8000"
        assert settings.observability.log_level == "WARNING"

    def test_extra_fields_forbidden(self) -> None:
        """RuntimeSettings rejects unknown fields."""
        # RuntimeSettings converts ValidationError to SettingsError
        with pytest.raises(SettingsError):
            RuntimeSettings(
                unknown_field="value"
            )  # RuntimeSettings.__init__ accepts **overrides: object

    def test_invalid_type_raises_settings_error(self) -> None:
        """RuntimeSettings raises SettingsError for invalid types."""
        with pytest.raises(SettingsError, match="Configuration validation failed"):  # type: ignore[call-arg]  # pytest.raises supports match, mypy stub issue
            RuntimeSettings(observability={"metrics_port": "not-an-int"})


class TestLoadSettings:
    """Tests for load_settings function."""

    def test_load_from_env(self) -> None:
        """load_settings loads from environment variables."""
        os.environ.pop("KGFOUNDRY_SEARCH__API_URL", None)
        settings = load_settings()
        assert isinstance(settings, RuntimeSettings)
        assert settings.search.api_url is None

    def test_load_with_overrides(self) -> None:
        """load_settings accepts programmatic overrides."""
        settings = load_settings(search={"api_url": "http://override:8000"})
        assert settings.search.api_url == "http://override:8000"

    def test_invalid_settings_raises_settings_error(self) -> None:
        """load_settings raises SettingsError for invalid settings."""
        with pytest.raises(SettingsError):
            load_settings(observability={"metrics_port": "not-an-int"})


class TestSettingsError:
    """Tests for SettingsError and Problem Details conversion."""

    def test_settings_error_has_problem_details(self) -> None:
        """SettingsError converts to Problem Details."""
        error = SettingsError(
            "Missing required environment variable",
            context={"env_var": "KGFOUNDRY_API_KEY"},
        )
        problem: ProblemDetails = error.to_problem_details(instance="urn:kgfoundry:settings:load")
        # TypedDict doesn't support isinstance, check structure instead
        assert problem["status"] == 500
        assert problem["type"] == "https://kgfoundry.dev/problems/configuration-error"
        assert "Missing required environment variable" in problem["detail"]
        assert problem["instance"] == "urn:kgfoundry:settings:load"
        # Context is mapped to errors field, not extensions
        errors = problem.get("errors")
        if errors is not None:
            assert isinstance(errors, dict)
            env_var = errors.get("env_var")
            assert isinstance(env_var, str)
            assert env_var == "KGFOUNDRY_API_KEY"

    def test_settings_error_preserves_cause(self) -> None:
        """SettingsError preserves cause chain."""
        cause = ValueError("Invalid type")
        error = SettingsError("Configuration failed", cause=cause)
        assert error.__cause__ is cause


class TestKgFoundrySettings:
    """Tests for KgFoundrySettings type alias."""

    def test_is_runtime_settings(self) -> None:
        """KgFoundrySettings is an alias for RuntimeSettings."""
        settings: KgFoundrySettings = RuntimeSettings()
        assert isinstance(settings, RuntimeSettings)


class TestNestedConfigs:
    """Tests for nested configuration models."""

    def test_sparse_embedding_defaults(self) -> None:
        """SparseEmbeddingConfig uses correct defaults."""
        config = SparseEmbeddingConfig()
        assert config.bm25_index_dir == "./_indices/bm25"
        assert config.bm25_k1 == 0.9
        assert config.bm25_b == 0.4
        assert config.splade_index_dir == "./_indices/splade_impact"
        assert config.splade_query_encoder == "naver/splade-v3-distilbert"

    def test_faiss_defaults(self) -> None:
        """FaissConfig uses correct defaults."""
        config = FaissConfig()
        assert config.gpu is True
        assert config.cuvs is True
        assert config.index_factory == "OPQ64,IVF8192,PQ64"
        assert config.nprobe == 64
        assert config.index_path == "./_indices/faiss/shard_000.idx"

    def test_nested_env_override(self) -> None:
        """Nested configurations can be overridden via environment."""
        os.environ["KGFOUNDRY_SPARSE_EMBEDDING__BM25_K1"] = "1.2"
        os.environ["KGFOUNDRY_FAISS__GPU"] = "false"
        try:
            settings = RuntimeSettings()
            assert settings.sparse_embedding.bm25_k1 == 1.2
            assert settings.faiss.gpu is False
        finally:
            os.environ.pop("KGFOUNDRY_SPARSE_EMBEDDING__BM25_K1", None)
            os.environ.pop("KGFOUNDRY_FAISS__GPU", None)


class TestRoundTrip:
    """Tests for settings round-trip validation."""

    def test_settings_can_be_recreated(self) -> None:
        """Settings can be recreated from their values."""
        settings1 = RuntimeSettings()
        settings2 = RuntimeSettings(
            search={"api_url": settings1.search.api_url, "k": settings1.search.k},
            observability={
                "log_level": settings1.observability.log_level,
                "metrics_enabled": settings1.observability.metrics_enabled,
            },
        )
        assert settings1.search.api_url == settings2.search.api_url
        assert settings1.search.k == settings2.search.k
        assert settings1.observability.log_level == settings2.observability.log_level
