"""Exceptions utilities."""

from __future__ import annotations

from typing import Final

from kgfoundry_common.errors import DownloadError as _DownloadError
from kgfoundry_common.errors import UnsupportedMIMEError as _UnsupportedMIMEError
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
class DownloadError(_DownloadError):
    """Compatibility alias for :class:`kgfoundry_common.errors.DownloadError`."""


# [nav:anchor UnsupportedMIMEError]
class UnsupportedMIMEError(_UnsupportedMIMEError):
    """Compatibility alias for :class:`kgfoundry_common.errors.UnsupportedMIMEError`."""
