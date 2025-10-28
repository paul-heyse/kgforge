"""Overview of config.

This module bundles config logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
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
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@kgfoundry-common",
        "stability": "stable",
        "since": "0.1.0",
    },
    "symbols": {
        "load_config": {
            "owner": "@kgfoundry-common",
            "stability": "stable",
            "since": "0.1.0",
        },
    },
}


# [nav:anchor load_config]
def load_config(path: str) -> dict[str, Any]:
    """Compute load config.

    Carry out the load config operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.
    
    Parameters
    ----------
    path : str
        Description for ``path``.
    
    Returns
    -------
    collections.abc.Mapping
        Description of return value.
    
    Examples
    --------
    >>> from kgfoundry_common.config import load_config
    >>> result = load_config(...)
    >>> result  # doctest: +ELLIPSIS
    ...
    """
    
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
