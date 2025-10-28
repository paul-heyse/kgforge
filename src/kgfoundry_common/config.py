"""Config utilities."""

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
    """Compute load config.

    Carry out the load config operation.

    Parameters
    ----------
    path : str
        Description for ``path``.

    Returns
    -------
    Mapping[str, Any]
        Description of return value.
    """
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




















    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
