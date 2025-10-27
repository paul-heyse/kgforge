"""Legacy exception aliases maintained for backwards compatibility.

NavMap:
- DownloadError: Raised when an external download fails.
- UnsupportedMIMEError: Raised when a MIME type is unsupported.
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
    """Raised when an external download fails."""

    ...


# [nav:anchor UnsupportedMIMEError]
class UnsupportedMIMEError(Exception):
    """Raised when a MIME type is unsupported."""

    ...
