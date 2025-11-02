"""Configuration helpers shared across kgfoundry.

This module provides typed configuration management via pydantic_settings.BaseSettings,
with automatic environment variable parsing, validation, and caching.

Examples
--------
>>> from kgfoundry_common.config import load_config
>>> settings = load_config()
>>> settings.log_level  # doctest: +SKIP
'INFO'
"""

from __future__ import annotations

import functools
from functools import lru_cache
from typing import Final

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from kgfoundry_common.logging import get_logger
from kgfoundry_common.navmap_types import NavMap
from kgfoundry_common.types import JsonPrimitive, JsonValue

__all__ = ["AppSettings", "JsonPrimitive", "JsonValue", "load_config"]

logger = get_logger(__name__)

__navmap__: Final[NavMap] = {
    "title": "kgfoundry_common.config",
    "synopsis": "Typed configuration management via pydantic_settings",
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
        "AppSettings": {
            "owner": "@kgfoundry-common",
            "stability": "stable",
            "since": "0.1.0",
        },
        "JsonPrimitive": {
            "owner": "@kgfoundry-common",
            "stability": "stable",
            "since": "0.1.0",
        },
        "JsonValue": {
            "owner": "@kgfoundry-common",
            "stability": "stable",
            "since": "0.1.0",
        },
        "load_config": {
            "owner": "@kgfoundry-common",
            "stability": "stable",
            "since": "0.1.0",
        },
    },
}


class AppSettings(BaseSettings):
    """Application settings with environment variable support.

    This class uses pydantic_settings to load and validate configuration
    from environment variables. All fields are immutable (frozen=True) and
    self-documenting via Field descriptions.

    Examples
    --------
    >>> import os
    >>> # Set environment variable
    >>> os.environ["LOG_LEVEL"] = "DEBUG"
    >>> settings = AppSettings()
    >>> settings.log_level
    'DEBUG'

    >>> # Validate invalid log level
    >>> os.environ["LOG_LEVEL"] = "INVALID"
    >>> try:
    ...     AppSettings()  # doctest: +SKIP
    ... except Exception:
    ...     print("Validation failed for invalid log level")
    """

    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        validation_alias="LOG_LEVEL",
    )

    log_format: str = Field(
        default="json",
        description="Logging format ('json' or 'text')",
        validation_alias="LOG_FORMAT",
    )

    model_config = {"frozen": True, "case_sensitive": False}

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        """Validate log level is one of the standard levels.

        Parameters
        ----------
        value : str
            Log level to validate.

        Returns
        -------
        str
            Validated log level (uppercase).

        Raises
        ------
        ValueError
            If log level is not valid.
        """
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        level_upper = value.upper()
        if level_upper not in valid_levels:
            msg = f"Invalid log level: {value}. Must be one of {valid_levels}"
            raise ValueError(msg)
        return level_upper

    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, value: str) -> str:
        """Validate log format is one of the supported formats.

        Parameters
        ----------
        value : str
            Log format to validate.

        Returns
        -------
        str
            Validated log format (lowercase).

        Raises
        ------
        ValueError
            If log format is not supported.
        """
        valid_formats = {"json", "text"}
        format_lower = value.lower()
        if format_lower not in valid_formats:
            msg = f"Invalid log format: {value}. Must be one of {valid_formats}"
            raise ValueError(msg)
        return format_lower


def _load_config_impl() -> AppSettings:
    """Load and validate application configuration from environment.

    Returns
    -------
    AppSettings
        Validated application settings.

    Raises
    ------
    ValueError
        If configuration validation fails.
    """
    try:
        settings = AppSettings()
        logger.log_success(
            "Configuration loaded",
            operation="config.load",
            extra={"log_level": settings.log_level, "log_format": settings.log_format},
        )
    except ValueError as exc:
        logger.log_failure(
            "Configuration validation failed",
            exception=exc,
            operation="config.load",
        )
        raise
    else:
        return settings


_load_config_cached: functools._lru_cache_wrapper[AppSettings] = lru_cache(maxsize=1)(
    _load_config_impl
)


def load_config(reload: bool = False) -> AppSettings:
    """Load application configuration from environment variables.

    This function caches the configuration to avoid repeated environment
    variable parsing. Pass reload=True to force a fresh load.

    Parameters
    ----------
    reload : bool, optional
        If True, clears the cache and reloads configuration from environment.
        Defaults to ``False``.

    Returns
    -------
    AppSettings
        Validated application settings.

    Raises
    ------
    ValueError
        If configuration validation fails (e.g., invalid log level).

    Examples
    --------
    >>> settings = load_config()
    >>> settings.log_level  # doctest: +SKIP
    'INFO'

    >>> # Reload configuration
    >>> settings = load_config(reload=True)  # doctest: +SKIP
    """
    if reload:
        _load_config_cached.cache_clear()

    return _load_config_cached()
