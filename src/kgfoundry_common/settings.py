"""Runtime settings with typed configuration and fail-fast validation.

This module provides RuntimeSettings (pydantic_settings.BaseSettings) with
nested models, environment variable support, and fail-fast Problem Details
on validation errors.

Examples
--------
>>> from kgfoundry_common.settings import RuntimeSettings
>>> # Fails fast if required env vars are missing
>>> settings = RuntimeSettings()  # Raises ConfigurationError if required vars missing
>>> assert settings.search_api_url is not None
"""

from __future__ import annotations

from typing import Final

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from kgfoundry_common.errors import SettingsError
from kgfoundry_common.logging import get_logger
from kgfoundry_common.navmap_types import NavMap

__all__ = [
    "FaissConfig",
    "KgFoundrySettings",
    "ObservabilityConfig",
    "RuntimeSettings",
    "SearchConfig",
    "SparseEmbeddingConfig",
    "load_settings",
]

logger = get_logger(__name__)

__navmap__: Final[NavMap] = {
    "title": "kgfoundry_common.settings",
    "synopsis": "Typed runtime configuration with fail-fast validation",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@kgfoundry-common",
        "stability": "stable",
        "since": "0.1.0",
    },
    "symbols": {
        name: {
            "owner": "@kgfoundry-common",
            "stability": "stable",
            "since": "0.1.0",
        }
        for name in __all__
    },
}


class SearchConfig(BaseSettings):
    """Search service configuration.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    _case_sensitive : bool | None, optional
        Describe ``_case_sensitive``.
        Defaults to ``None``.
    _nested_model_default_partial_update : bool | None, optional
        Describe ``_nested_model_default_partial_update``.
        Defaults to ``None``.
    _env_prefix : str | None, optional
        Describe ``_env_prefix``.
        Defaults to ``None``.
    _env_file : DotenvType | None, optional
        Describe ``_env_file``.
        Defaults to ``PosixPath('.')``.
    _env_file_encoding : str | None, optional
        Describe ``_env_file_encoding``.
        Defaults to ``None``.
    _env_ignore_empty : bool | None, optional
        Describe ``_env_ignore_empty``.
        Defaults to ``None``.
    _env_nested_delimiter : str | None, optional
        Describe ``_env_nested_delimiter``.
        Defaults to ``None``.
    _env_nested_max_split : int | None, optional
        Describe ``_env_nested_max_split``.
        Defaults to ``None``.
    _env_parse_none_str : str | None, optional
        Describe ``_env_parse_none_str``.
        Defaults to ``None``.
    _env_parse_enums : bool | None, optional
        Describe ``_env_parse_enums``.
        Defaults to ``None``.
    _cli_prog_name : str | None, optional
        Describe ``_cli_prog_name``.
        Defaults to ``None``.
    _cli_parse_args : bool | list[str] | tuple[str, ...] | None, optional
        Describe ``_cli_parse_args``.
        Defaults to ``None``.
    _cli_settings_source : CliSettingsSource[Any] | None, optional
        Describe ``_cli_settings_source``.
        Defaults to ``None``.
    _cli_parse_none_str : str | None, optional
        Describe ``_cli_parse_none_str``.
        Defaults to ``None``.
    _cli_hide_none_type : bool | None, optional
        Describe ``_cli_hide_none_type``.
        Defaults to ``None``.
    _cli_avoid_json : bool | None, optional
        Describe ``_cli_avoid_json``.
        Defaults to ``None``.
    _cli_enforce_required : bool | None, optional
        Describe ``_cli_enforce_required``.
        Defaults to ``None``.
    _cli_use_class_docs_for_groups : bool | None, optional
        Describe ``_cli_use_class_docs_for_groups``.
        Defaults to ``None``.
    _cli_exit_on_error : bool | None, optional
        Describe ``_cli_exit_on_error``.
        Defaults to ``None``.
    _cli_prefix : str | None, optional
        Describe ``_cli_prefix``.
        Defaults to ``None``.
    _cli_flag_prefix_char : str | None, optional
        Describe ``_cli_flag_prefix_char``.
        Defaults to ``None``.
    _cli_implicit_flags : bool | None, optional
        Describe ``_cli_implicit_flags``.
        Defaults to ``None``.
    _cli_ignore_unknown_args : bool | None, optional
        Describe ``_cli_ignore_unknown_args``.
        Defaults to ``None``.
    _cli_kebab_case : bool | None, optional
        Describe ``_cli_kebab_case``.
        Defaults to ``None``.
    _cli_shortcuts : Mapping[str, str | list[str]] | None, optional
        Describe ``_cli_shortcuts``.
        Defaults to ``None``.
    _secrets_dir : PathType | None, optional
        Describe ``_secrets_dir``.
        Defaults to ``None``.
    api_url : str | NoneType, optional
        Search API base URL. Required if search features are used.
        Environment variable: `KGFOUNDRY_SEARCH_API_URL`.
        Defaults to None.
        Defaults to ``None``.
    k : int, optional
        Default number of results to return.
        Environment variable: `KGFOUNDRY_SEARCH_K`.
        Defaults to 10.
        Defaults to ``10``.
    dense_candidates : int, optional
        Number of dense candidates for hybrid search.
        Environment variable: `KGFOUNDRY_SEARCH_DENSE_CANDIDATES`.
        Defaults to 200.
        Defaults to ``200``.
    sparse_candidates : int, optional
        Number of sparse candidates for hybrid search.
        Environment variable: `KGFOUNDRY_SEARCH_SPARSE_CANDIDATES`.
        Defaults to 200.
        Defaults to ``200``.
    rrf_k : int, optional
        Reciprocal Rank Fusion parameter.
        Environment variable: `KGFOUNDRY_SEARCH_RRF_K`.
        Defaults to 60.
        Defaults to ``60``.
    sparse_backend : str, optional
        Sparse backend type ('lucene' or 'pure').
        Environment variable: `KGFOUNDRY_SEARCH_SPARSE_BACKEND`.
        Defaults to 'lucene'.
        Defaults to ``'lucene'``.
    kg_boosts_direct : float, optional
        Direct KG boost weight.
        Environment variable: `KGFOUNDRY_SEARCH_KG_BOOSTS_DIRECT`.
        Defaults to 0.08.
        Defaults to ``0.08``.
    kg_boosts_one_hop : float, optional
        One-hop KG boost weight.
        Environment variable: `KGFOUNDRY_SEARCH_KG_BOOSTS_ONE_HOP`.
        Defaults to 0.04.
        Defaults to ``0.04``.
    validate_responses : bool, optional
        Enable response schema validation (dev/staging only).
        Environment variable: `SEARCH_API_VALIDATE`.
        Defaults to False.
        Defaults to ``False``.
    """

    model_config = SettingsConfigDict(env_prefix="KGFOUNDRY_SEARCH_", extra="forbid")

    api_url: str | None = Field(
        default=None,
        description="Search API base URL (required if search features are used)",
    )
    k: int = Field(default=10, description="Default number of results to return")
    dense_candidates: int = Field(
        default=200, description="Number of dense candidates for hybrid search"
    )
    sparse_candidates: int = Field(
        default=200, description="Number of sparse candidates for hybrid search"
    )
    rrf_k: int = Field(default=60, description="Reciprocal Rank Fusion parameter")
    sparse_backend: str = Field(
        default="lucene", description="Sparse backend type ('lucene' or 'pure')"
    )
    kg_boosts_direct: float = Field(default=0.08, description="Direct KG boost weight")
    kg_boosts_one_hop: float = Field(default=0.04, description="One-hop KG boost weight")
    validate_responses: bool = Field(
        default=False, description="Enable response schema validation (dev/staging only)"
    )


class ObservabilityConfig(BaseSettings):
    """Observability configuration.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    _case_sensitive : bool | None, optional
        Describe ``_case_sensitive``.
        Defaults to ``None``.
    _nested_model_default_partial_update : bool | None, optional
        Describe ``_nested_model_default_partial_update``.
        Defaults to ``None``.
    _env_prefix : str | None, optional
        Describe ``_env_prefix``.
        Defaults to ``None``.
    _env_file : DotenvType | None, optional
        Describe ``_env_file``.
        Defaults to ``PosixPath('.')``.
    _env_file_encoding : str | None, optional
        Describe ``_env_file_encoding``.
        Defaults to ``None``.
    _env_ignore_empty : bool | None, optional
        Describe ``_env_ignore_empty``.
        Defaults to ``None``.
    _env_nested_delimiter : str | None, optional
        Describe ``_env_nested_delimiter``.
        Defaults to ``None``.
    _env_nested_max_split : int | None, optional
        Describe ``_env_nested_max_split``.
        Defaults to ``None``.
    _env_parse_none_str : str | None, optional
        Describe ``_env_parse_none_str``.
        Defaults to ``None``.
    _env_parse_enums : bool | None, optional
        Describe ``_env_parse_enums``.
        Defaults to ``None``.
    _cli_prog_name : str | None, optional
        Describe ``_cli_prog_name``.
        Defaults to ``None``.
    _cli_parse_args : bool | list[str] | tuple[str, ...] | None, optional
        Describe ``_cli_parse_args``.
        Defaults to ``None``.
    _cli_settings_source : CliSettingsSource[Any] | None, optional
        Describe ``_cli_settings_source``.
        Defaults to ``None``.
    _cli_parse_none_str : str | None, optional
        Describe ``_cli_parse_none_str``.
        Defaults to ``None``.
    _cli_hide_none_type : bool | None, optional
        Describe ``_cli_hide_none_type``.
        Defaults to ``None``.
    _cli_avoid_json : bool | None, optional
        Describe ``_cli_avoid_json``.
        Defaults to ``None``.
    _cli_enforce_required : bool | None, optional
        Describe ``_cli_enforce_required``.
        Defaults to ``None``.
    _cli_use_class_docs_for_groups : bool | None, optional
        Describe ``_cli_use_class_docs_for_groups``.
        Defaults to ``None``.
    _cli_exit_on_error : bool | None, optional
        Describe ``_cli_exit_on_error``.
        Defaults to ``None``.
    _cli_prefix : str | None, optional
        Describe ``_cli_prefix``.
        Defaults to ``None``.
    _cli_flag_prefix_char : str | None, optional
        Describe ``_cli_flag_prefix_char``.
        Defaults to ``None``.
    _cli_implicit_flags : bool | None, optional
        Describe ``_cli_implicit_flags``.
        Defaults to ``None``.
    _cli_ignore_unknown_args : bool | None, optional
        Describe ``_cli_ignore_unknown_args``.
        Defaults to ``None``.
    _cli_kebab_case : bool | None, optional
        Describe ``_cli_kebab_case``.
        Defaults to ``None``.
    _cli_shortcuts : Mapping[str, str | list[str]] | None, optional
        Describe ``_cli_shortcuts``.
        Defaults to ``None``.
    _secrets_dir : PathType | None, optional
        Describe ``_secrets_dir``.
        Defaults to ``None``.
    log_level : str, optional
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        Environment variable: `KGFOUNDRY_LOG_LEVEL`.
        Defaults to "INFO".
        Defaults to ``'INFO'``.
    metrics_enabled : bool, optional
        Enable Prometheus metrics export.
        Environment variable: `KGFOUNDRY_METRICS_ENABLED`.
        Defaults to True.
        Defaults to ``True``.
    traces_enabled : bool, optional
        Enable OpenTelemetry tracing.
        Environment variable: `KGFOUNDRY_TRACES_ENABLED`.
        Defaults to False.
        Defaults to ``False``.
    metrics_port : int, optional
        Port for Prometheus metrics endpoint.
        Environment variable: `KGFOUNDRY_METRICS_PORT`.
        Defaults to 9090.
        Defaults to ``9090``.
    """

    model_config = SettingsConfigDict(env_prefix="KGFOUNDRY_", extra="forbid")

    log_level: str = Field(
        default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    metrics_enabled: bool = Field(default=True, description="Enable Prometheus metrics export")
    traces_enabled: bool = Field(default=False, description="Enable OpenTelemetry tracing")
    metrics_port: int = Field(default=9090, description="Port for Prometheus metrics endpoint")


class SparseEmbeddingConfig(BaseSettings):
    """Sparse embedding configuration.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    _case_sensitive : bool | None, optional
        Describe ``_case_sensitive``.
        Defaults to ``None``.
    _nested_model_default_partial_update : bool | None, optional
        Describe ``_nested_model_default_partial_update``.
        Defaults to ``None``.
    _env_prefix : str | None, optional
        Describe ``_env_prefix``.
        Defaults to ``None``.
    _env_file : DotenvType | None, optional
        Describe ``_env_file``.
        Defaults to ``PosixPath('.')``.
    _env_file_encoding : str | None, optional
        Describe ``_env_file_encoding``.
        Defaults to ``None``.
    _env_ignore_empty : bool | None, optional
        Describe ``_env_ignore_empty``.
        Defaults to ``None``.
    _env_nested_delimiter : str | None, optional
        Describe ``_env_nested_delimiter``.
        Defaults to ``None``.
    _env_nested_max_split : int | None, optional
        Describe ``_env_nested_max_split``.
        Defaults to ``None``.
    _env_parse_none_str : str | None, optional
        Describe ``_env_parse_none_str``.
        Defaults to ``None``.
    _env_parse_enums : bool | None, optional
        Describe ``_env_parse_enums``.
        Defaults to ``None``.
    _cli_prog_name : str | None, optional
        Describe ``_cli_prog_name``.
        Defaults to ``None``.
    _cli_parse_args : bool | list[str] | tuple[str, ...] | None, optional
        Describe ``_cli_parse_args``.
        Defaults to ``None``.
    _cli_settings_source : CliSettingsSource[Any] | None, optional
        Describe ``_cli_settings_source``.
        Defaults to ``None``.
    _cli_parse_none_str : str | None, optional
        Describe ``_cli_parse_none_str``.
        Defaults to ``None``.
    _cli_hide_none_type : bool | None, optional
        Describe ``_cli_hide_none_type``.
        Defaults to ``None``.
    _cli_avoid_json : bool | None, optional
        Describe ``_cli_avoid_json``.
        Defaults to ``None``.
    _cli_enforce_required : bool | None, optional
        Describe ``_cli_enforce_required``.
        Defaults to ``None``.
    _cli_use_class_docs_for_groups : bool | None, optional
        Describe ``_cli_use_class_docs_for_groups``.
        Defaults to ``None``.
    _cli_exit_on_error : bool | None, optional
        Describe ``_cli_exit_on_error``.
        Defaults to ``None``.
    _cli_prefix : str | None, optional
        Describe ``_cli_prefix``.
        Defaults to ``None``.
    _cli_flag_prefix_char : str | None, optional
        Describe ``_cli_flag_prefix_char``.
        Defaults to ``None``.
    _cli_implicit_flags : bool | None, optional
        Describe ``_cli_implicit_flags``.
        Defaults to ``None``.
    _cli_ignore_unknown_args : bool | None, optional
        Describe ``_cli_ignore_unknown_args``.
        Defaults to ``None``.
    _cli_kebab_case : bool | None, optional
        Describe ``_cli_kebab_case``.
        Defaults to ``None``.
    _cli_shortcuts : Mapping[str, str | list[str]] | None, optional
        Describe ``_cli_shortcuts``.
        Defaults to ``None``.
    _secrets_dir : PathType | None, optional
        Describe ``_secrets_dir``.
        Defaults to ``None``.
    bm25_index_dir : str, optional
        BM25 index directory path.
        Environment variable: `KGFOUNDRY_SPARSE_EMBEDDING_BM25_INDEX_DIR`.
        Defaults to './_indices/bm25'.
        Defaults to ``'./_indices/bm25'``.
    bm25_k1 : float, optional
        BM25 k1 parameter.
        Environment variable: `KGFOUNDRY_SPARSE_EMBEDDING_BM25_K1`.
        Defaults to 0.9.
        Defaults to ``0.9``.
    bm25_b : float, optional
        BM25 b parameter.
        Environment variable: `KGFOUNDRY_SPARSE_EMBEDDING_BM25_B`.
        Defaults to 0.4.
        Defaults to ``0.4``.
    splade_index_dir : str, optional
        SPLADE index directory path.
        Environment variable: `KGFOUNDRY_SPARSE_EMBEDDING_SPLADE_INDEX_DIR`.
        Defaults to './_indices/splade_impact'.
        Defaults to ``'./_indices/splade_impact'``.
    splade_query_encoder : str, optional
        SPLADE query encoder model name.
        Environment variable: `KGFOUNDRY_SPARSE_EMBEDDING_SPLADE_QUERY_ENCODER`.
        Defaults to 'naver/splade-v3-distilbert'.
        Defaults to ``'naver/splade-v3-distilbert'``.
    """

    model_config = SettingsConfigDict(env_prefix="KGFOUNDRY_SPARSE_EMBEDDING_", extra="forbid")

    bm25_index_dir: str = Field(default="./_indices/bm25", description="BM25 index directory path")
    bm25_k1: float = Field(default=0.9, description="BM25 k1 parameter")
    bm25_b: float = Field(default=0.4, description="BM25 b parameter")
    splade_index_dir: str = Field(
        default="./_indices/splade_impact", description="SPLADE index directory path"
    )
    splade_query_encoder: str = Field(
        default="naver/splade-v3-distilbert",
        description="SPLADE query encoder model name",
    )


class FaissConfig(BaseSettings):
    """FAISS index configuration.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    _case_sensitive : bool | None, optional
        Describe ``_case_sensitive``.
        Defaults to ``None``.
    _nested_model_default_partial_update : bool | None, optional
        Describe ``_nested_model_default_partial_update``.
        Defaults to ``None``.
    _env_prefix : str | None, optional
        Describe ``_env_prefix``.
        Defaults to ``None``.
    _env_file : DotenvType | None, optional
        Describe ``_env_file``.
        Defaults to ``PosixPath('.')``.
    _env_file_encoding : str | None, optional
        Describe ``_env_file_encoding``.
        Defaults to ``None``.
    _env_ignore_empty : bool | None, optional
        Describe ``_env_ignore_empty``.
        Defaults to ``None``.
    _env_nested_delimiter : str | None, optional
        Describe ``_env_nested_delimiter``.
        Defaults to ``None``.
    _env_nested_max_split : int | None, optional
        Describe ``_env_nested_max_split``.
        Defaults to ``None``.
    _env_parse_none_str : str | None, optional
        Describe ``_env_parse_none_str``.
        Defaults to ``None``.
    _env_parse_enums : bool | None, optional
        Describe ``_env_parse_enums``.
        Defaults to ``None``.
    _cli_prog_name : str | None, optional
        Describe ``_cli_prog_name``.
        Defaults to ``None``.
    _cli_parse_args : bool | list[str] | tuple[str, ...] | None, optional
        Describe ``_cli_parse_args``.
        Defaults to ``None``.
    _cli_settings_source : CliSettingsSource[Any] | None, optional
        Describe ``_cli_settings_source``.
        Defaults to ``None``.
    _cli_parse_none_str : str | None, optional
        Describe ``_cli_parse_none_str``.
        Defaults to ``None``.
    _cli_hide_none_type : bool | None, optional
        Describe ``_cli_hide_none_type``.
        Defaults to ``None``.
    _cli_avoid_json : bool | None, optional
        Describe ``_cli_avoid_json``.
        Defaults to ``None``.
    _cli_enforce_required : bool | None, optional
        Describe ``_cli_enforce_required``.
        Defaults to ``None``.
    _cli_use_class_docs_for_groups : bool | None, optional
        Describe ``_cli_use_class_docs_for_groups``.
        Defaults to ``None``.
    _cli_exit_on_error : bool | None, optional
        Describe ``_cli_exit_on_error``.
        Defaults to ``None``.
    _cli_prefix : str | None, optional
        Describe ``_cli_prefix``.
        Defaults to ``None``.
    _cli_flag_prefix_char : str | None, optional
        Describe ``_cli_flag_prefix_char``.
        Defaults to ``None``.
    _cli_implicit_flags : bool | None, optional
        Describe ``_cli_implicit_flags``.
        Defaults to ``None``.
    _cli_ignore_unknown_args : bool | None, optional
        Describe ``_cli_ignore_unknown_args``.
        Defaults to ``None``.
    _cli_kebab_case : bool | None, optional
        Describe ``_cli_kebab_case``.
        Defaults to ``None``.
    _cli_shortcuts : Mapping[str, str | list[str]] | None, optional
        Describe ``_cli_shortcuts``.
        Defaults to ``None``.
    _secrets_dir : PathType | None, optional
        Describe ``_secrets_dir``.
        Defaults to ``None``.
    gpu : bool, optional
        Enable GPU acceleration.
        Environment variable: `KGFOUNDRY_FAISS_GPU`.
        Defaults to True.
        Defaults to ``True``.
    cuvs : bool, optional
        Enable cuVS support.
        Environment variable: `KGFOUNDRY_FAISS_CUVS`.
        Defaults to True.
        Defaults to ``True``.
    index_factory : str, optional
        FAISS index factory string.
        Environment variable: `KGFOUNDRY_FAISS_INDEX_FACTORY`.
        Defaults to 'OPQ64,IVF8192,PQ64'.
        Defaults to ``'OPQ64,IVF8192,PQ64'``.
    nprobe : int, optional
        Number of probes for IVF indexes.
        Environment variable: `KGFOUNDRY_FAISS_NPROBE`.
        Defaults to 64.
        Defaults to ``64``.
    index_path : str, optional
        FAISS index file path.
        Environment variable: `KGFOUNDRY_FAISS_INDEX_PATH`.
        Defaults to './_indices/faiss/shard_000.idx'.
        Defaults to ``'./_indices/faiss/shard_000.idx'``.
    """

    model_config = SettingsConfigDict(env_prefix="KGFOUNDRY_FAISS_", extra="forbid")

    gpu: bool = Field(default=True, description="Enable GPU acceleration")
    cuvs: bool = Field(default=True, description="Enable cuVS support")
    index_factory: str = Field(
        default="OPQ64,IVF8192,PQ64", description="FAISS index factory string"
    )
    nprobe: int = Field(default=64, description="Number of probes for IVF indexes")
    index_path: str = Field(
        default="./_indices/faiss/shard_000.idx", description="FAISS index file path"
    )


class RuntimeSettings(BaseSettings):
    """Runtime configuration with typed nested models and fail-fast validation.

    <!-- auto:docstring-builder v1 -->

    This class loads configuration from environment variables with type validation.
    Missing required fields raise SettingsError with Problem Details metadata.

    Parameters
    ----------
    search : SearchConfig, optional
        Describe ``search``.
        Defaults to ``<factory>``.
    observability : ObservabilityConfig, optional
        Describe ``observability``.
        Defaults to ``<factory>``.
    sparse_embedding : SparseEmbeddingConfig, optional
        Describe ``sparse_embedding``.
        Defaults to ``<factory>``.
    faiss : FaissConfig, optional
        Describe ``faiss``.
        Defaults to ``<factory>``.

    Environment Variables
    ---------------------
    KGFOUNDRY_SEARCH_API_URL : str, optional
        Search API base URL (required if search features are used).
    KGFOUNDRY_SEARCH_K : int, optional
        Default number of search results (default: 10).
    KGFOUNDRY_SEARCH_SPARSE_BACKEND : str, optional
        Sparse backend type ('lucene' or 'pure', default: 'lucene').
    KGFOUNDRY_SPARSE_EMBEDDING_BM25_INDEX_DIR : str, optional
        BM25 index directory (default: './_indices/bm25').
    KGFOUNDRY_FAISS_GPU : bool, optional
        Enable FAISS GPU acceleration (default: True).
    KGFOUNDRY_LOG_LEVEL : str, optional
        Logging level (default: INFO).
    KGFOUNDRY_METRICS_ENABLED : bool, optional
        Enable Prometheus metrics (default: True).
    KGFOUNDRY_TRACES_ENABLED : bool, optional
        Enable OpenTelemetry tracing (default: False).

    Examples
    --------
    >>> import os
    >>> os.environ["KGFOUNDRY_SEARCH_API_URL"] = "http://localhost:8000"
    >>> settings = RuntimeSettings()
    >>> assert settings.search.api_url == "http://localhost:8000"

    Raises
    ------
    SettingsError
        If required environment variables are missing or invalid.
        The error includes Problem Details metadata for structured error handling.

    Notes
    -----
    - All nested models use `extra="forbid"` to prevent unknown fields
    - Settings are validated at instantiation time (fail-fast)
    - Configuration can be overridden via environment variables or `.env` files
    """

    model_config = SettingsConfigDict(
        env_prefix="KGFOUNDRY_",
        env_nested_delimiter="__",
        extra="forbid",
        case_sensitive=False,
    )

    search: SearchConfig = Field(
        default_factory=SearchConfig, description="Search service configuration"
    )
    observability: ObservabilityConfig = Field(
        default_factory=ObservabilityConfig, description="Observability configuration"
    )
    sparse_embedding: SparseEmbeddingConfig = Field(
        default_factory=SparseEmbeddingConfig,
        description="Sparse embedding configuration",
    )
    faiss: FaissConfig = Field(default_factory=FaissConfig, description="FAISS index configuration")

    def __init__(self, **overrides: object) -> None:
        """Initialize settings with fail-fast validation from environment variables.

        <!-- auto:docstring-builder v1 -->

        This constructor loads from environment variables (via pydantic_settings)
        and accepts optional keyword arguments for programmatic overrides. For
        public API usage, prefer `load_settings()` which provides better error
        handling.

        Parameters
        ----------
        **overrides : object
            Optional keyword arguments to override default or environment values.
            Keys must match field names (e.g., `search`, `observability`).

        Raises
        ------
        SettingsError
            If validation fails (missing required fields, invalid types, etc.).
        """
        try:
            super().__init__(**overrides)  # type: ignore[arg-type]  # BaseSettings.__init__ accepts Any kwargs, mypy can't infer overloads
        except Exception as exc:
            # Convert Pydantic validation errors to SettingsError with Problem Details
            msg = f"Configuration validation failed: {exc}"
            logger.exception(
                "Settings validation failed",
                extra={"error": str(exc), "error_type": type(exc).__name__},
            )
            raise SettingsError(
                msg,
                cause=exc,
                context={"validation_error": str(exc)},
            ) from exc


# Type alias for backward compatibility
KgFoundrySettings = RuntimeSettings


def load_settings(**overrides: object) -> KgFoundrySettings:
    """Load settings from environment variables with optional overrides.

    <!-- auto:docstring-builder v1 -->

    This function loads settings from environment variables (via pydantic_settings)
    and allows programmatic overrides via keyword arguments. All overrides are
    validated against the settings schema.

    Parameters
    ----------
    **overrides : object
        Optional keyword arguments to override default or environment values.
        Keys must match field names in RuntimeSettings (e.g., `search`, `observability`).

    Returns
    -------
    RuntimeSettings
        Validated settings instance.

    Raises
    ------
    SettingsError
        If required environment variables are missing or validation fails.
        The error includes Problem Details metadata for structured error handling.

    Examples
    --------
    >>> import os
    >>> os.environ.pop("KGFOUNDRY_SEARCH_API_URL", None)
    >>> # Load from environment only
    >>> settings = load_settings()
    >>> assert settings.search.api_url is None
    >>>
    >>> # Override specific values
    >>> settings = load_settings(search={"api_url": "http://localhost:8000"})
    >>> assert settings.search.api_url == "http://localhost:8000"
    >>>
    >>> # Missing required env var raises SettingsError
    >>> # settings = load_settings()  # doctest: +SKIP
    >>> # SettingsError: Configuration validation failed
    """
    try:
        return RuntimeSettings(**overrides)
    except SettingsError:
        # Re-raise SettingsError as-is
        raise
    except Exception as exc:
        # Convert any other validation errors to SettingsError
        msg = f"Failed to load settings: {exc}"
        logger.exception(
            "Settings loading failed",
            extra={"error": str(exc), "error_type": type(exc).__name__},
        )
        raise SettingsError(
            msg,
            cause=exc,
            context={"validation_error": str(exc)},
        ) from exc
