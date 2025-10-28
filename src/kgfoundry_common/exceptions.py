"""Provide utilities for module.

Notes
-----
This module exposes the primary interfaces for the package.

See Also
--------
kgfoundry_common.exceptions
"""


from __future__ import annotations

from typing import Final

from kgfoundry_common.navmap_types import NavMap

__all__ = ["DownloadError", "UnsupportedMIMEError"]

__navmap__: Final[NavMap] = {
    "title": "kgfoundry_common.exceptions",
    "synopsis": "Legacy exception aliases maintained for backwards compatibility",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": ["DownloadError", "UnsupportedMIMEError"],
        },
    ],
}


# [nav:anchor DownloadError]
class DownloadError(Exception):
    """Describe DownloadError."""
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    ...


# [nav:anchor UnsupportedMIMEError]
class UnsupportedMIMEError(Exception):
    """Describe UnsupportedMIMEError."""
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    ...
