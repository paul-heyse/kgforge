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
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@kgfoundry-common",
        "stability": "deprecated",
        "since": "0.1.0",
        "deprecated_in": "0.3.0",
    },
    "symbols": {
        "DownloadError": {
            "owner": "@kgfoundry-common",
            "stability": "deprecated",
            "since": "0.1.0",
            "deprecated_in": "0.3.0",
        },
        "UnsupportedMIMEError": {
            "owner": "@kgfoundry-common",
            "stability": "deprecated",
            "since": "0.1.0",
            "deprecated_in": "0.3.0",
        },
    },
}
# [nav:anchor DownloadError]
DownloadError = _DownloadError
DownloadError.__doc__ = (
    "Compatibility alias for :class:`kgfoundry_common.errors.DownloadError`."
)


# [nav:anchor UnsupportedMIMEError]
UnsupportedMIMEError = _UnsupportedMIMEError
UnsupportedMIMEError.__doc__ = (
    "Compatibility alias for :class:`kgfoundry_common.errors.UnsupportedMIMEError`."
)
