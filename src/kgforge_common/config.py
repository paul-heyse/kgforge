"""Module for kgforge_common.config.

NavMap:
- load_config: Load config.
"""

from __future__ import annotations

from typing import Any

import yaml


def load_config(path: str) -> dict[str, Any]:
    """Load config.

    Parameters
    ----------
    path : str
        TODO.

    Returns
    -------
    Dict[str, Any]
        TODO.
    """
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
