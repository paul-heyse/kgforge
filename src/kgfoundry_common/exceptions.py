"""Module for kgfoundry_common.exceptions.

NavMap:
- DownloadError: Raised when an external download fails.
- UnsupportedMIMEError: Raised when a MIME type is unsupported.
"""


class DownloadError(Exception):
    """Raised when an external download fails."""

    ...


class UnsupportedMIMEError(Exception):
    """Raised when a MIME type is unsupported."""

    ...
