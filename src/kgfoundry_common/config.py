"""Module for kgfoundry_common.config.

NavMap:
- NavMap: Structure describing a module navmap.
- load_config: Load a YAML configuration file from ``path``.
"""

from __future__ import annotations

from typing import Any, Final

import yaml

from kgfoundry_common.navmap_types import NavMap

__all__ = ["load_config"]

__navmap__: Final[NavMap] = {
    "title": "kgfoundry_common.config",
    "synopsis": "Configuration helpers shared across kgfoundry",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": ["load_config"],
        },
    ],
}


# [nav:anchor load_config]
def load_config(path: str) -> dict[str, Any]:
    """Load a YAML configuration file from ``path``.

    Parameters
    ----------
    path : str
        Path to the configuration file on disk.

    Returns
    -------
    dict[str, Any]
        Parsed configuration values.
    """
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
