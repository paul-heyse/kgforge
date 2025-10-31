"""Configuration helpers shared across kgfoundry.

This module provides YAML configuration loading and type aliases for JSON values
used across configuration modules. Type aliases are imported from problem_details
for consistency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import yaml

from kgfoundry_common.navmap_types import NavMap

# Import JSON type aliases from problem_details for consistency
from kgfoundry_common.problem_details import JsonPrimitive, JsonValue

__all__ = ["JsonPrimitive", "JsonValue", "load_config"]

__navmap__: Final[NavMap] = {
    "title": "kgfoundry_common.config",
    "synopsis": "Configuration helpers shared across kgfoundry",
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


# [nav:anchor load_config]
def load_config(path: str) -> dict[str, object]:
    """Load configuration from a YAML file.

    <!-- auto:docstring-builder v1 -->

        This function loads and validates YAML configuration files, ensuring
        all keys are strings and the root value is a dictionary.

        Parameters
        ----------
        path : str
            Path to YAML configuration file.

        Returns
        -------
        dict[str, object]
            Parsed configuration dictionary with string keys.

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist.
        TypeError
            If the YAML does not decode to a dictionary or contains non-string keys.

        Examples
        --------
        >>> import tempfile
        >>> import os
        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        ...     _ = f.write("key1: value1
    key2: 42
    ")
        ...     config_path = f.name
        >>> config = load_config(config_path)
        >>> assert config["key1"] == "value1"
        >>> assert config["key2"] == 42
        >>> os.unlink(config_path)

    Parameters
    ----------
    path : str
        Describe ``path``.

    Returns
    -------
    dict[str, object]
        Describe return value.
"""
    with Path(path).open(encoding="utf-8") as f:
        loaded: object = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        message = f"Configuration at {path} must decode to a mapping"
        raise TypeError(message)
    validated: dict[str, object] = {}
    for key, value in loaded.items():
        if not isinstance(key, str):
            message = f"Configuration keys must be strings; received {key!r}"
            raise TypeError(message)
        validated[key] = value
    return validated
