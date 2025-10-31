"""Configuration helpers shared across kgfoundry.

This module provides YAML configuration loading and type aliases for JSON values
used across configuration modules. Type aliases are imported from problem_details
for consistency.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import yaml

from kgfoundry_common.navmap_types import NavMap

# Import JSON type aliases from problem_details for consistency
from kgfoundry_common.problem_details import JsonPrimitive, JsonValue

if TYPE_CHECKING:
    from typing import Any

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
def load_config(path: str) -> dict[str, Any]:
    """Describe load config.

    <!-- auto:docstring-builder v1 -->

    Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

    Parameters
    ----------
    path : str
        Describe ``path``.


    Returns
    -------
    dict[str, Any]
        Describe return value.












    Raises
    ------
    TypeError
    Raised when TODO for TypeError.
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
